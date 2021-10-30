#include <iostream>
#include <vector>
#include <omp.h>
#include <initializer_list>
#include <random>
#include <chrono>
#include "gauss.h"
#include "Matrix.h"

using namespace std;

#define TEST_BASIC
//#define CALCULATE_ERROR
#define TEST_LOWER
//#define TEST_UPPER
//#define PRINT_MATRIX

uniform_real_distribution<float> distr(-1'000, 1'000);
uniform_int_distribution<int> mid_distr(10'000, 100'000);

void test(int n, int m, mt19937& my_rand)
{


#ifdef TEST_LOWER
	triangle_matrix<double, false> a_lower(n);

	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			a_lower[i][j] = distr(my_rand);
		}
		a_lower[i][i] = mid_distr(my_rand);
		if (mid_distr(my_rand) % 2) {
			a_lower[i][i] = -a_lower[i][i];
		}
	}
#endif	
#ifdef TEST_UPPER
	triangle_matrix<double, true> a_upper(n);

	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = i + 1; j < n; j++)
		{
			a_upper[i][j] = distr(my_rand);
		}
		a_upper[i][i] = mid_distr(my_rand);
		if (mid_distr(my_rand) % 2) {
			a_upper[i][i] = -a_upper[i][i];
		}
	}
#endif


	matrix_columns<double> b(n, m);

	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			b[i][j] = distr(my_rand);
		}
	}

	auto begin = chrono::high_resolution_clock::now();
	auto end = chrono::high_resolution_clock::now();
	auto elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

#ifdef TEST_LOWER
	begin = chrono::high_resolution_clock::now();

	auto res_lower_block = solve_lower_system_blocks<decltype(a_lower), decltype(b), matrix<double>>(a_lower, b, n, m);

	end = chrono::high_resolution_clock::now();
	elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "block algorithm for lower system time: " << elapsed_secs << "ms" << std::endl;

#endif
#ifdef TEST_UPPER

	begin = chrono::high_resolution_clock::now();

	auto res_upper_block = solve_upper_system_blocks(a_upper, b, n, m);

	end = chrono::high_resolution_clock::now();
	elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "block algorithm for upper system time: " << elapsed_secs << "ms" << std::endl;
#endif

#ifdef TEST_BASIC
#ifdef TEST_LOWER
	begin = chrono::high_resolution_clock::now();

	auto res_lower_basic = calc_L<decltype(a_lower), decltype(b), matrix<double>>(a_lower, b, n, m);

	end = chrono::high_resolution_clock::now();
	elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "basic algorithm for lower system time: " << elapsed_secs << "ms" << std::endl;
#endif
#ifdef TEST_UPPER
	begin = chrono::high_resolution_clock::now();

	auto res_upper_basic = calc_U(a_upper, b, n, m);

	end = chrono::high_resolution_clock::now();
	elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "basic algorithm for upper system time: " << elapsed_secs << "ms" << std::endl;

#endif
#endif

	cout << '\n';

#ifdef CALCULATE_ERROR
#ifdef TEST_BASIC
#ifdef TEST_LOWER
	double max_error_lower_basic = 0;
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			double s = 0;
			for (size_t k = 0; k <= i; k++)
			{
				s += a_lower[i][k] * res_lower_basic[k][j];
			}

			if (abs(b[i][j]) > 1e-8) {
				max_error_lower_basic = max(max_error_lower_basic, abs(s - b[i][j]) / abs(b[i][j]));
			}
		}
	}

	cout << "basic algorithm for lower system error : " << max_error_lower_basic << endl;
#endif
#ifdef TEST_UPPER
	double max_error_upper_basic = 0;
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			double s = 0;
			for (size_t k = i; k < n; k++)
			{
				s += a_upper[i][k] * res_upper_basic[k][j];
			}

			if (abs(b[i][j]) > 1e-8) {
				max_error_upper_basic = max(max_error_upper_basic, abs(s - b[i][j]) / abs(b[i][j]));
			}
		}
	}

	cout << "basic algorithm for upper system error : " << max_error_upper_basic << endl;
#endif
#endif
#endif

#ifdef CALCULATE_ERROR
#ifdef TEST_LOWER
	double max_error_lower_block = 0;
	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			double s = 0;
			for (size_t k = 0; k <= i; k++)
			{
				s += a_lower[i][k] * res_lower_block[k][j];
			}

			if (abs(b[i][j]) > 1e-8) {
				max_error_lower_block = max(max_error_lower_block, abs(s - b[i][j]) / abs(b[i][j]));
			}
		}
	}

	cout << "block algorithm for lower system error: : " << max_error_lower_block << endl;
#endif
#ifdef TEST_UPPER
	double max_error_upper_block = 0;
	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			double s = 0;
			for (size_t k = i; k < n; k++)
			{
				s += a_upper[i][k] * res_upper_block[k][j];
			}

			if (abs(b[i][j]) > 1e-8) {
				max_error_upper_block = max(max_error_upper_block, abs(s - b[i][j]) / abs(b[i][j]));
			}
		}
	}

	cout << "block algorithm for upper system error: : " << max_error_upper_block << endl;
#endif
#endif
}

int main()
{

	size_t seed = random_device{}();
	//seed = 3349713019;

	cout << seed << endl << endl;

	mt19937 my_rand(seed);
	
	test(3000, 2000, my_rand);




	
#ifdef PRINT_MATRIX

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j <= i; j++)
		{
			cout << a_lower[i][j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
	
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			cout << b[i][j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			cout << res_lower_block[i][j] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
#endif
	
	return 0;
}
