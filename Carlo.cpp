#include<stdio.h>
// classes example
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>


using namespace std;

class MonteCarloHost 
{
public:
    // Constants to be altered
    int N_iter=1000;
    int N_dim = 5;
    int N_walkers=10;

    // Set functions
    void set_dimensions (int x, int y, int z) {N_iter = x; N_walkers = y; N_dim = z;}; 

    // Reporting function
    void print_settings() 
    {
        cout << "\nIterations : "<< N_iter << "\nWalkers : " << N_walkers << "\nDimensions: " << N_dim;

        // Report step sizes
        int i;
        cout << "\n-------------------";
        cout << "\nStep sizes\n-------------------";
        for (i=0;i<N_dim;i++) cout << "\nNdim " << i+1 << " : " << StepSize[i];
        cout << "\n-------------------\n";

    };

    // Handle the chain length
    double * chain = (double*)malloc(N_iter*N_walkers*N_dim*sizeof(double));
    double * llike = (double*)malloc(N_iter*N_walkers*sizeof(double));
    double * StepSize = (double*)malloc(N_dim*sizeof(double));

    void reallocate() 
    {
        chain = (double *)realloc(chain, N_iter*N_walkers*N_dim*sizeof(double));
        llike = (double *)realloc(llike, N_iter*N_walkers*sizeof(double));
        StepSize = (double *)realloc(StepSize, N_dim*sizeof(double));
    };

    void SetStepSize(double * step)
    {
        for (int i=0; i < N_dim; i++) StepSize[i] = step[i];
    };
    void sampler_MH(double * p0);


    double loglike();



    // Random number generators
    double rand_normal(double mean, double stddev);
    double rand_uniform(double min, double max);


};


void MonteCarloHost::sampler_MH(double * p0)
{
    // Declarations
    int i,j;

    // First set p0 in the chain
    for (i=0; i<N_dim*N_walkers; i++) chain[i] = p0[i]; 

    // Now evalute the first step 
    for (i=0; i < N_walkers; i++) llike[i]  = loglike();

    // Now run the sampler
    for (i=1; i < N_iter; i++)
    {
        // Now cycle walkers
        // this is where openMP would operate over the walkers
        //#pragma omp parallel for
        for (j=0; j < N_walkers; j++)
        {
            // Create a trial step from the last
            double trial_step[N_dim];
            int k,l;
            for (k=0; k < N_dim; k++) trial_step[k] = rand_normal(chain[i*N_dim*j + k], StepSize[k]);


            // Now get the new loglike
            for (k=0; k < N_dim; k++) llike[i*N_dim + j] =  loglike();

            // Now asses if its good or not
            if (1.0 > exp(llike[(i-1)*N_dim + j] - llike[i*N_dim + j]) ) //rand_uniform(0.0,1.0))
            {
                // This is a better step
                // Update the step
                cout << "\nbetter " << llike[(i-1)*N_dim + j] << " " << llike[i*N_dim + j] <<  " " <<exp(llike[(i-1)*N_dim + j] - llike[i*N_dim + j]);
                for (k=0; k < N_dim; k++) chain[i*N_dim*j + k] = trial_step[k];
            }
            else
            {
                // This is a worser step
                cout << "\nworse " << llike[(i-1)*N_dim + j] << " " << llike[i*N_dim + j] <<  " " << exp(llike[(i-1)*N_dim + j] - llike[i*N_dim + j]);
                for (k=0; k < N_dim; k++) chain[i*N_dim*j + k] = chain[(i-1)*N_dim*j + k];
                llike[i*N_dim + j] = llike[(i-1)*N_dim + j];
            }
        }
        chain[(i-1)*N_dim*N_walkers ];
    }  
}




double MonteCarloHost::rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}


double MonteCarloHost::rand_uniform(double min, double max)
{
    return ((double) rand() / (RAND_MAX+1)) * (max-min+1) + min;
}




double MonteCarloHost::loglike()
{
    return rand_normal(1.0,0.01);
}



int main()
{
    // MCMC parameters
    int N_iter = 100000;
    int N_walkers = 10;
    int N_dim = 5;
    int i;

    // Seed the random number generator
    srand((unsigned)time(NULL));


    // Declare the host
    class MonteCarloHost mcmc;
    mcmc.set_dimensions(N_iter, N_walkers, N_dim);
    mcmc.reallocate();


    double p0[N_walkers*N_dim];
    for (i=0; i<N_dim*N_walkers; i++) p0[i] = abs(mcmc.rand_normal(0.0,0.6));
    double step[N_dim];
    for (i=0; i<N_dim*N_walkers; i++) step[i] =mcmc.rand_normal(1.0,0.01);
    mcmc.SetStepSize(step);
    mcmc.print_settings();


    mcmc.sampler_MH(p0);

}
