import subprocess
from itertools import count
from queue import Queue
from threading import Thread

import click
import pandas as pd


def gen_bound_pairs(size, num_threads):
    batch_size = size // num_threads + 1
    bounds = list(range(0, size, batch_size)) + [size]
    return list(zip(bounds[:-1], bounds[1:]))


def worker(task_queue):
    """Worker function to process tasks from the queue"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        model_path, dataset_path, output_folder, bounds, batch_size, gpu = task
        command = [
            'python', 'gen_dpo_dataset.py',
            '--model-path', model_path,
            '--dataset-path', dataset_path,
            '--output-folder', output_folder,
            '--prompts-bounds', bounds,
            '--batch-size', batch_size,
            '--gpu', str(gpu),
            #'--extra',
        ]
        print(' '.join(command))
        subprocess.run(command)
        task_queue.task_done()


@click.command()
@click.option('--num-threads', default=2, help='The number of threads i.e. number of parallely running models.')
@click.option('--batch-size', default=50, help='The size of parquet file to save.')
def run_parallel_evaluation(num_threads, batch_size):
    datasets = [
        ['../../data/VCTK-Corpus', '../../data/VCTK-Corpus_gen'],
        ['../../data/keithito_lj_speech', '../../data/keithito_lj_speech_gen'],
    ]
    models = [
        'vctk-asr',
        'lg-asr',
    ]

    # Create a queue to hold all tasks
    task_queue = Queue()

    dataset_size = [
        [len(pd.read_csv(d + '/metadata.csv')) for d in data] for data in datasets
    ]

    bounds = [
        [gen_bound_pairs(d_s, num_threads) for d_s in data_size] for data_size in dataset_size
    ]

    suffixes = ['', '-gen']
    num_tasks = 0
    gpu_counter = count()
    # Create all tasks and put them in the queue
    for model_name, dataset, dataset_bounds in zip(models, datasets, bounds):
        for data, _bounds, suffix in zip(dataset, dataset_bounds, suffixes):
            for bound in _bounds:
                task = (
                    f'../../checkpoints/finale_models/{model_name}',
                    data,
                    f'../../data/dpo_dataset/{model_name}{suffix}',
                    f'{bound[0]},{bound[1]}',
                    f'{batch_size}',
                    next(gpu_counter) % 8,
                )
                task_queue.put(task)
                num_tasks += 1

    print(f'{num_tasks}'.center(60, '='))

    # Create 2 worker threads

    threads = []
    for _ in range(num_threads):
        thread = Thread(target=worker, args=(task_queue,))
        thread.start()
        threads.append(thread)

    # Wait for all tasks to complete
    task_queue.join()

    # Stop workers
    for _ in range(num_threads):
        task_queue.put(None)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    run_parallel_evaluation()
