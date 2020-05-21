//Importing Necessary Library

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <cmath>
#include<ctime>


#pragma comment (lib, "msmpi.lib")

using namespace std;
using namespace std::chrono;

void Matrix_Multiplication(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size);
void Matrix_Multiplication_Parallel(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size);
void Matrix_Multiplication_Parallel_Transposed(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size);
void Block_Matrix_Multiplication_Parallel(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size, int block_size);

void output_matrix(long **matrix, int size);

int main(int argc, char **argv)
{
	clock_t start,end;
	int size, block_size;
	cout<<"The Order of Matrix Multiplication is A*B \n";
	cout<<"The Dimension for the Matrix is Same N*N \n";
	cout<<"Enter the Dimension of Matrix \n";
	cin >> size;
	cout<<"Enter the Block Size \n";
	cin>>block_size;
	cout<<"\n";
	long **Matrix_A = new long*[size];
	long **Matrix_B = new long*[size];
	long **Matrix_BT = new long*[size];
	long **Matrix_C = new long*[size];
	long **Matrix_CC = new long*[size];
	int max = 100;
    srand((unsigned) time(0));
	cout << "Matrix Initialization is in progress \n";
	cout<<"\n";
	
	for (int i = 0; i < size; i++)
	{
		Matrix_A[i] = new long[size];
		Matrix_B[i] = new long[size];
		Matrix_BT[i] = new long[size];
		Matrix_C[i] = new long[size];
		Matrix_CC[i] = new long[size];
		for (int j = 0; j < size; j++)
		{
			Matrix_A[i][j] = (rand()%max)+1;
			Matrix_B[i][j] = (rand()%max)+1;
			Matrix_C[i][j] = 0;
			Matrix_CC[i][j] = 0;
		}
	}
	
	for (int it = 0; it < 1; it++)
	{
		
		// Noraml Matrix Multiplication 
		cout << "Normal Matrix Multiplication Begin \n";
		start=clock();
        Matrix_Multiplication(Matrix_A, Matrix_B, Matrix_C, size);
		end=clock();
        cout<<"Execution Time :"<<(double (end-start))/CLOCKS_PER_SEC<<"\n";
		cout<<"\n";
		
		
		
		// Paralle Matrix Multiplication using opneMP
		cout << "Parallel Matrix Multiplication using OpenMP Begin \n";
        start=clock();
		Matrix_Multiplication_Parallel(Matrix_A, Matrix_B, Matrix_C, size);
        end=clock();
        cout<<"Execution Time :"<<(double (end-start))/CLOCKS_PER_SEC<<"\n";
        cout<<"\n";



		// Paralle Matrix Multiplication using OpenMP and Transpose of B  
		cout << "Parallel Matrix Multiplicatoin using OpenMP and Transpose of B Begin \n";
        start=clock();
		Matrix_Multiplication_Parallel_Transposed(Matrix_A, Matrix_BT, Matrix_C, size);
        end=clock();
        cout<<"Execution Time:"<<(double (end-start))/CLOCKS_PER_SEC<<endl;
		cout<<"\n";



		
		//Paralle Matrix Multiplication using Blocks and OpenMP
		cout << "Parallel Matrix Multiplication using Blocks and OpenMP Begin \n";
		start=clock();
		Block_Matrix_Multiplication_Parallel(Matrix_A, Matrix_B, Matrix_CC, size, block_size);
		end=clock();
	    cout<<"Execution Time:"<<(double (end-start))/CLOCKS_PER_SEC<<endl;
		cout<<"\n";
	}
	return 0;
}


void Matrix_Multiplication(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				Matrix_C[i][j] += Matrix_A[i][k] * Matrix_B[k][j];
			}
		}
	}	
}

void Matrix_Multiplication_Parallel(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size)
{
	int i,j,k;
	int thread_id;
	int chunk = 100;
	
	//Critical Section
	
	#pragma omp parallel shared(Matrix_A, Matrix_B, Matrix_C, size, chunk) private(i, j, k, thread_id)
	{
		thread_id = omp_get_thread_num();
		if (thread_id == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}	
		#pragma omp for schedule (dynamic,chunk)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				for (k = 0; k < size; k++)
				{
					Matrix_C[i][j] += Matrix_A[i][k] * Matrix_B[k][j];
				}
			}
		}
	}
     
    //Critical section end	
}


void Matrix_Multiplication_Parallel_Transposed(long **Matrix_A, long **Matrix_BT, long **Matrix_C, int size)
{
	
	int i,j,k;
	int thread_id;
	int chunk =100;
	 
	// Critical Section begin
	 
	#pragma omp parallel shared(Matrix_A, Matrix_BT, Matrix_C, size,chunk) private(i, j, k, thread_id)
	{
		thread_id = omp_get_thread_num();
		if (thread_id == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}
	#pragma omp for schedule(dynamic)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				for (k = 0; k < size; k++)
				{
					Matrix_C[i][j] += Matrix_A[i][k] * Matrix_BT[j][k];
				}
			}
		}
	}
	// Critical Section end
}

void Block_Matrix_Multiplication_Parallel(long **Matrix_A, long **Matrix_B, long **Matrix_C, int size, int block_size)
{
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	int num;
	//int chunk = 1;
	int thread_id;
	
	//Critical Section
#pragma omp parallel shared(Matrix_A, Matrix_B, Matrix_C, size) private(i, j, k, jj, kk, thread_id, num)
	{
		thread_id = omp_get_thread_num();
		if (thread_id == 0)
		{
			cout << "Number of threads: " << omp_get_num_threads() << endl;
		}
		#pragma omp for schedule(dynamic)
		for (jj = 0; jj < size; jj += block_size)
		{
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						num = 0;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							num += Matrix_A[i][k] * Matrix_B[k][j];
						}
						Matrix_C[i][j] += num;
					}
				}
			}
		}
	}
	// Critical Section
}

void output_matrix(long **matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << "\n";
	}
}
