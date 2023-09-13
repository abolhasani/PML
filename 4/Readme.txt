Hi,
This is the readme file required for my (Sadegh Abolhasani, u1416052) HW4 for CS 6190 Spring 23.

I have provided as much comment as possible. I coded both problems in Python.
I run the codes on Google Colab and I have copied them to .py files here, attached to this compressed file.
I upload the data to colab and this is my main method of data input. 

P1: I asked a TA for help, and I completely changed all the code for section C. Also I verified the 
initialization parameters with him. After the office hour still a problem existed and that was because I was
updating m before s. After that, the problem of having different answers for different iterations was solved.
I copied the gauss-hermite function from the example code.

P2: For variable transformation and still working with the provided gauss-hermite function from before, 
I added the arguments mean and var to a nwe function called shifted_sigmoid and used it prior to integration. 
the vlr function for getting fully factorised posterior don't seem to be ok (after a long time spent on debugging)
and I expected this to work better than the others, not the worst.

P3: Nothing special. One problem that I had with both problems 1 and 3 was that I had forgotton to keep the 
initial parameters and they were updating and unaccessible. I added seperate variables for updating and
after that the issue no longer existed (on the first iteration, I ise the initial and then switch
to use updated ones)

P4: Didn't do
