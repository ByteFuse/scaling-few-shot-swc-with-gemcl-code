from argparse import ArgumentParser
import csv
import os
from torch.utils.data import DataLoader
from meta_dataset import MetaCommonWordsSingleLanguage as MetaDataset
from model.gemcl import GeMCL
from einops import rearrange
import torch
from glob import glob
from os import path
from utils import prepare_mfccs, write_up_experiment_data
import pandas as pd
from loguru import logger
from tqdm import trange
import time


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config ={
    'tasks': 1000,
    'train_shots':5,
    'test_shots':5,
    'batch_size':10,
    'eval_batch_size':10,
    'log_dir':'./checkpoints/meta_train_wav2vec2/25way_5shot',
    'sub_tasks': 25,
    'episode_bite':1,
}

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

config['seed'] = args.seed

def meta_test(config: dict):

    torch.manual_seed(config['seed']) # Setting the seed
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])

    model = GeMCL(config, continual_learning=True)
    model = model.to(device)
    model.eval()

    # Load the latest checkpoint

    checkpoint_files = sorted(glob(path.join(config['log_dir'], 'step_*.pt')))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    meta_test_path = f"../../meta_splits/train_{config['train_shots']}_shots_test_{config['test_shots']}_shots/en/meta_test_classes.txt"
    meta_train_path = f"../../meta_splits/train_{config['train_shots']}_shots_test_{config['test_shots']}_shots/en/meta_train_classes.txt"

    meta_test_set = MetaDataset('en', '../../data/macroset', 'test', config, meta_train_path, meta_test_path, '../../splits')

    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        collate_fn=meta_test_set.episode_collate_fn)
    meta_test_loader_iter = iter(meta_test_loader)

    os.makedirs(config['log_dir'], exist_ok=True)
    num_lessons = config['tasks'] // config['sub_tasks'] # the number of lessons each episode has.

    sub_test_samples = config['sub_tasks'] * config['test_shots'] # the number of samples to be tested on.
    sub_train_samples = config['sub_tasks'] * config['train_shots'] # the number of samples to be learned.

    task_experimental_results = []
    class_experimental_results = []
    total_samples_seen = 0

    # Begin Meta-Testing
    # * Batsi: I apologise for the monsterous code you are about to witness.

    task_result_csv_path = f"./experiments/meta-test/en_{config['tasks']}-way-{config['train_shots']}-shot/task_experimental_results.csv"
    class_result_csv_path = f"./experiments/meta-test/en_{config['tasks']}-way-{config['train_shots']}-shot/class_experimental_results.csv"
    learn_class_stats_times_path = f"./experiments/meta-test/en_{config['tasks']}-way-{config['train_shots']}-shot/learn_class_stats_times.csv"
    os.makedirs(os.path.dirname(task_result_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(class_result_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(learn_class_stats_times_path), exist_ok=True)

    with open(learn_class_stats_times_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Lesson', 'Time_to_Learn_Seconds', "MFCC_Prep_Seconds", "Total_Lesson_Block_Seconds","Classes_Learned"])


    with torch.no_grad():
        train_x, _, test_x, test_y = next(meta_test_loader_iter) # Comes in shape (batch, tasks*shots, ...)
        write_up_experiment_data(config, train_x, test_x, "meta-test", single_language=True, language="en")

        # * The batch is essentially the number of episodes. So we can iterate over the episodes. 
        # * And each episode is N-way K-shot episode. N is config['tasks'] and K is config['train_shots'].
        # * Batsi: Not very happy with this part. Needs refactor. Potentially we have to change the dataloader for evaluation.
        # * Batsi: We plan to do large batch sizes, i.e. large episodes during evaluation so we have to keep this in mind and we plan to run multiple episodes.

        for episode in trange(len(train_x)):
            # * For each episode we learn a set of classes (sub-tasks) incrementally. We call this a lesson. We learn sub_train_samples each lesson.
            for lesson in range(num_lessons):

                start_time_lesson = time.perf_counter()
                train_x_current_task = train_x[episode][total_samples_seen:total_samples_seen + sub_train_samples]

                start_time_mfcc_prep = time.perf_counter()
                train_x_current_task = prepare_mfccs([train_x_current_task])
                end_time_mfcc_prep = time.perf_counter()

                mfcc_prep_time = end_time_mfcc_prep - start_time_mfcc_prep

                train_x_current_task = train_x_current_task.to(device)

                # * Meeasuring the time taken to learn the class statistics for logging purposes.
                # Read this to see why time.perf_counter() is used: https://builtin.com/articles/timing-functions-python
                start_time = time.perf_counter()
                model.learn_class_statistics(train_x_current_task)
                end_time = time.perf_counter()
                time_to_learn = end_time - start_time

                total_samples_seen += sub_train_samples
                classes_learned = total_samples_seen // config['train_shots']
                samples_to_test = classes_learned * config['test_shots']

                end_time_lesson = time.perf_counter()
                lesson_time = end_time_lesson - start_time_lesson

                with open(learn_class_stats_times_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, lesson, f"{time_to_learn:.6f}", f"{mfcc_prep_time:.6f}", f"{lesson_time:.6f}", classes_learned])


                # * After learning the current set of classes, we evaluate on all seen classes so far
                for test_idx in range(0, samples_to_test, sub_test_samples):

                    start_test_time = time.perf_counter()
                    test_x_current_task = test_x[episode][test_idx:test_idx + sub_test_samples]
                    test_y_current_task = test_y[episode][test_idx:test_idx + sub_test_samples]
                    test_y_current_task = rearrange(test_y_current_task, '(t s) -> 1 (t s)', t=config['sub_tasks'], s=config['test_shots']) # reshape to input into the model.

                    start_time_mfcc_prep = time.perf_counter()
                    test_x_current_task = prepare_mfccs([test_x_current_task])
                    end_time_mfcc_prep = time.perf_counter()
                    mfcc_prep_time = end_time_mfcc_prep - start_time_mfcc_prep

                    test_x_current_task = test_x_current_task.to(device)
                    test_y_current_task = test_y_current_task.to(device)

                    time_start_inference = time.perf_counter()
                    loss, (acc, class_acc) = model.test_forward(test_x_current_task, test_y_current_task)
                    time_end_inference = time.perf_counter()
                    inference_time = time_end_inference - time_start_inference

                    labels_list = test_y_current_task[0, ::config['test_shots']].cpu().tolist()
                    accuracies_list = class_acc[0].cpu().tolist()
                    end_time_test = time.perf_counter()

                    test_time = end_time_test - start_test_time
                    
                    # avoiding another nested for loop here.
                    # Got the idea from here: https://www.geeksforgeeks.org/python/create-a-pandas-dataframe-from-lists/
                    class_result_df = pd.DataFrame({
                                        "episode": [episode]*len(labels_list),
                                        "lesson": [lesson]*len(labels_list),
                                        "classes_learned": [classes_learned]*len(labels_list),
                                        "class_label": labels_list,
                                        "class_accuracy": accuracies_list
                                    })
                    class_experimental_results.append(class_result_df) 

                    task_result_row = {
                                        "episode": episode,
                                        "lesson": lesson,
                                        "task_id": test_idx // sub_test_samples,
                                        "classes_learned": classes_learned,
                                        "accuracy": acc.item(),
                                        "loss": loss.item(),
                                        "inference_time": inference_time,
                                        "mfcc_prep_time": mfcc_prep_time,
                                        "test_time": test_time,
                                    }
                    task_experimental_results.append(task_result_row)



            # Save the results after each episode        
            task_df = pd.DataFrame(task_experimental_results)
            class_df = pd.concat(class_experimental_results, ignore_index=True)

            # write task data
            write_header = not os.path.exists(task_result_csv_path)
            task_df.to_csv(task_result_csv_path, mode='a', header=write_header, index=False)

            # write class data
            write_header = not os.path.exists(class_result_csv_path)
            class_df.to_csv(class_result_csv_path, mode='a', header=write_header, index=False)
            
            task_experimental_results = []
            class_experimental_results = []
            logger.info(f"Completed Episode {episode}")   
            total_samples_seen = 0 # reset for the next episode.
            model.reset_class_statistics() # reset the class statistics for the next episode.

meta_test(config)
