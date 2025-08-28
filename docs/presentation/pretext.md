# Presentation Script Draft

## Part 1: Introduction

Good morning/afternoon, professor. Thank you for being here.

My name is GL and I am a student in the Master of Data Science program.

Today, I am excited to present my thesis, titled: End-to-End PPO-Based Reinforcement Learning Framework for Scalable 0/1 Knapsack Problem Solving -- From Data Generation to Large-Scale Generalization.

My research focuses on finding a general ML method to solve 0/1 knapsack problem.

## Part 2: What is the 0/1 Knapsack Problem?

So, let's start with the core concept: the 0/1 Knapsack Problem.

To put it simply, imagine we have a backpack. This backpack has a limited capacity, for example, it can only hold 15 kilograms.

We also have a collection of items. Each item has its own weight and its own value.

The rule is, for any item, we can either put it in the backpack or leave it. We cannot take just a part of an item. This 'take it or leave it' choice is the '0-1' part of the name.

The main question is: How do we choose the items? Our goal is to maximize the total value of the items in our backpack, but without exceeding the capacity.

The knapsack problem is more than just a puzzle. It's a simple model for many real-world problems. For example, it's used in finance to create an investment portfolio. It's also used in computing to allocate resources.

The 0-1 version I just described is the simplest type of knapsack problem. There are more complex versions. For instance, some problems have multiple limits, multiple goals, or relationships between items.

Finally, the knapsack problem is a classic 'NP-complete' problem. In simple terms, this means that there is no universal, fast algorithm to find the perfect solution, especially for large-scale cases. This is why we study different methods to find good solutions."

## Part 3: How to Solve the Knapsack Problem?

"So, how do we solve this complex problem? The methods can be divided into two main groups: traditional algorithms and modern approaches.

First, let's talk about **traditional algorithms**.
There are exact algorithms, which always find the best solution. Examples are brute-force, dynamic programming, and branch-and-bound.
There are also approximate algorithms. These are mainly heuristics, like genetic algorithms or simulated annealing. They are usually faster but may not find the perfect solution.

More recently, researchers have turned to **modern approaches**, mainly using Machine Learning.
These ML methods can also be put into two categories.

First, there are **constructive methods**. These methods are end-to-end. They learn to build a solution directly from the problem description.

Second, there are **improvement methods**. These methods start with a solution that already exists, maybe a decent one. They then try to improve it. These methods often need to use commercial solvers.

This brings me to the current research landscape and the motivation for my work.

As we can see from this table, very few papers use Machine Learning to solve the knapsack problem. And the existing research has some important limitations.

Most of these methods can only solve problems of a **fixed size**. This means a model trained to solve a 50-item problem cannot solve a 100-item problem.

The one paper that can handle unfixed sizes had to use a commercial solver.

This is where my research comes in. My method is different and it addresses these gaps.

* **First, my framework can solve problems of variable sizes.**
* **Second, it generalizes well. We can train the model on small-scale problems, and it can then solve large-scale problems effectively.**
* **And finally, my method is truly end-to-end. It does not need any external commercial solvers.**

## Part 4: The Limits of Traditional Algorithms

"Now, let's dive deeper into the performance of the traditional exact algorithms and a commercial solver like Gurobi.

First, let's look at the figure on the left. As you can see, when the problem size gets bigger, the memory required for traditional exact algorithms explodes. Eventually, the program crashes because it runs out of memory.

You might notice that Dynamic Programming, or DP, is not even shown in this figure. That is because it is often the first method to fail. It consumes too much memory to even handle moderately sized problems.

Now, please look at the second figure. The time it takes to find a solution also grows exponentially with the problem size.

So, the key takeaway from these figures is clear: While traditional exact methods are accurate, they are simply not scalable. They struggle to handle large-scale knapsack problems."

