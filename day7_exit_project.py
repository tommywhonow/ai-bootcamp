from typing import List, Generator
from dataclasses import dataclass, field
import random
import time
import numpy as np
import statistics
from contextlib import contextmanager


@dataclass
class Student:
    name: str
    scores: list[float]

    def average(self) -> float:
        return float(np.mean(self.scores))
    
#Generator for students
def load_students(students: list[Student]) -> Generator[Student, None, None]:
    for student in students:
        yield student

#Normalise
def normalise(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std() # type: ignore[no-any-return]

#Top studnets
def top_students(students: list[Student], threshold: float) -> list[Student]:
    return[
        student
        for student in students
        if student.average() > threshold
    ]

@contextmanager
def pipeline_timer() -> Generator[None, None, None]:
    start = time.time()
    try:
        yield
    finally:
        end: float= time.time()
        print(f"It took {end - start: .4f} seconds")

#Generate scores
def generate_scores(count: int=3, low: int=50, high: int=100) -> list[float]:
    return[random.randint(low, high) for _ in range(count)]

#Define threshold
threshold =75

#run the whole pipeline
with pipeline_timer(): 
    names = ["Adam", "Bob", "Charlie", "Dave", "Eve"]
    students = [
        Student(name, generate_scores(3,50,100))
        for name in names
        ]
    #Generator
    student_list = list(load_students(students))
    #numpy
    scores_array = np.array(
        [i.scores for i in student_list],
        dtype=np.float64
        )
    
    print("All students: ")
    for i in student_list:
        print(f"{i.name}: {i.scores}"
              f"avg = {i.average(): .2f}")
        
    #normalise
    normalised_scores = normalise(scores_array)
      
    print("their normalisation score is : \n ", normalised_scores)
        
    top = top_students(student_list, threshold)
    x = [x.name for x in top]
    print("\nTop students: ", x)