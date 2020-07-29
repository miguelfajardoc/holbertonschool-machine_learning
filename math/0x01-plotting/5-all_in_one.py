#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.8)
plt.suptitle("All in one")

# line
graphic1 = plt.subplot(3, 2, 1)
graphic1.plot(y0, 'r-')
graphic1.axis([0, 10, None, None])
plt.yticks(np.arange(1001, step=500))

# scatter
graphic2 = plt.subplot(3, 2, 2)
graphic2.scatter(x1, y1, c="m", marker=".")
graphic2.set_title("Men's Height vs Weight", fontsize="x-small")
graphic2.set_ylabel('Weight (lbs)', fontsize="x-small")
graphic2.set_xlabel('Height (in)', fontsize="x-small")

# log
graphic3 = plt.subplot(3, 2, 3)
graphic3.plot(x2, y2)
graphic3.set_yscale('log')
graphic3.set_title('Exponential Decay of C-14', fontsize="x-small")
graphic3.set_ylabel('Fraction Remaining', fontsize="x-small")
graphic3.set_xlabel('Time (years)', fontsize="x-small")
graphic3.axis([0, 28650, None, None])

# two in one
graphic4 = plt.subplot(3, 2, 4)
graphic4.plot(x3, y31, '--r', label='C-14')
graphic4.plot(x3, y32, '-g', label='Ra-226')
graphic4.legend()
graphic4.set_xlabel('Time (years)', fontsize="x-small")
graphic4.set_ylabel('Fraction Remaining', fontsize="x-small")
graphic4.set_title('Exponential Decay of Radioactive Elements',
                   fontsize="x-small")
graphic4.axis([0, 20000, 0, 1])

# Histograma
graphic5 = plt.subplot(3, 1, 3)
grades = np.arange(101, step=10)
NOfStudents = np.arange(31, step=5)
plt.hist(student_grades, bins=grades, edgecolor='black', linewidth=1)
plt.xlabel('Grades', fontsize="x-small")
plt.ylabel('Number of Students', fontsize="x-small")
plt.title('Project A', fontsize="x-small")
plt.axis([0, 100, 0, 30])
plt.xticks(grades)
plt.yticks(NOfStudents)

plt.show()
