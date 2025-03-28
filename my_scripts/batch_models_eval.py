import subprocess
from threading import Thread
from queue import Queue
import os


def worker(task_queue):
    """Worker function to process tasks from the queue"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        dir, model, dataset = task
        subprocess.run([
            'python', 'evaluate_tts.py',
            '--model-name', f'{model}',
            '--dataset-path', dataset,
            '--model-path', f'../checkpoints/{dir}/{model}',
            '--output-name', f'{model}-metrics'
        ])
        task_queue.task_done()


def run_parallel_evaluation():
    models = [
        'base-vctk-dpo-best',
        'base-vctk-dpo-last',
    ]
    datasets = [
        # '../data/facebook_voxpopuli',
        # '../data/keithito_lj_speech',
        '../data/VCTK-Corpus',
    ]

    # Create a queue to hold all tasks
    task_queue = Queue()

    # Create all tasks and put them in the queue
    for dataset in datasets:
        for model in models:
            task_queue.put(('finale_models', model, dataset))

    # Create 2 worker threads
    num_threads = 2
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