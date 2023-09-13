Hi,
This is the readme file required for my (Sadegh Abolhasani, u1416052) HW5 for CS 6190 Spring 23.

I have provided as much comment as possible. I coded both problems in Python.
For this HW, I ran the first two codes at Google Colab and the other two on my VM at my home department.

P1: Nothing to mention. I used the functions and parts of the code later in the other problems

P2: For variable transformation and still working with the provided gauss-hermite function from before, 
I added the arguments mean and var to a nwe function called shifted_sigmoid and used it prior to integration. 
the vlr function for getting fully factorised posterior don't seem to be ok (after a long time spent on debugging)
and I expected this to work better than the others, not the worst.

P3: A: I used the H = U + K formula, and use the neccesary functions to implement the hmc. 
    I do the parts before and after burn-in in the same function. The overal look of the functions are borrowed from P1
    B: I do the parts before and after burn-in in the same function. I use truncate function from stats.
    for doing so, I had to use a lower and upper truncation points that have been computed according to scipy.
    for the predictive log-likelihood and accuracy, I have used the original formula provided in the question.
    I use a tolerance to handle numerical stability and avoid getting logs mistakenly of 0
    C: Didn't do
    D: Didn't do

P4: Didn't do
