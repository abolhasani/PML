Hi,
This is the readme file required for my (Sadegh Abolhasani, u1416052) HW2 for CS 6190 Spring 23.

I have provided as much comment as possible. I coded both problems in Python.
I run the codes on Google Colab and I have copied them to .py files here, attached to this compressed file.

P1: The main idea here was to create a grid to hold probability values.
Then, these values were used to print the heat map. There are a prior and posterior function.
These functions return the mean and covariance of prior and posterior that are used by other functions
to generate samples and compute the regression and print heat map and lines.

P2: For part B, I asked one of the TAs and she told me although it is expected to have lower accuracies, but my recorded
accuracy seems a bit off, so I'd better normalize my inputs. But I realized that my code had some
problems too. I both rewrote the code and implemented normalization. You can see my many tries that have been now commented.
For part C, I asked her again and she told me I can use the probit function that I had from part B
and use the gradient and the hessian computed in it to capture the Newton-Raphson update, and it didn't seem
that it needed any other modification. After implementation I checked with her again to see if the results make sense. 

P3: Didn't do
