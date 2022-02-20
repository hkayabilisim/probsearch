This is an incomplete implementation of probabilistic search
explained in this interesting article: 

    Schmidhuber, Jürgen. "Discovering neural nets with low Kolmogorov 
    complexity and high generalization capability." Neural Networks 10.5 (1997): 857-873.

My implementation is incomplete because I didn't calculate Levin complexity components.
I've just implemented the Universal Turing Machine in section 3.2.
When you run it, the probabilistic search will output some random programs that is capable 
of generating weights such that the machine learning problem shown in 4.3.1 is solved 
with perfect generalization capability (i.e work for all unseen testing 
samples.)

    {'trial': 135, 'cycle': 200, 'training': True, 'testing': True, 'program': (0)1,(1)1,(2)0,(3)2,(4)3,(5)0,}
    {'trial': 438, 'cycle': 300, 'training': True, 'testing': True, 'program': (0)1,(1)0,(2)0,(3)0,(4)0,(5)1,}
    {'trial': 603, 'cycle': 150, 'training': True, 'testing': True, 'program': (0)1,(1)1,(2)0,(3)1,(4)0,(5)1,}


