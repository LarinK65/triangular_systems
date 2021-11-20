#include <iostream>
#include <vector>
#include <omp.h>
#include <initializer_list>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include "gauss.h"
#include "Matrix.h"
#include "tester_funcs.h"

#include <mkl.h>


using namespace std;

using type = double;

void create_test(size_t n, size_t m, std::mt19937& my_rand, const std::string& file_descr, int mt, int64_t base_seed)
{
	matrix<double> a = (mt == 0 ? 
		generate_matrix<matrix<type>>(matrix_types::lower, n, n, my_rand) 
		: generate_matrix<matrix<type>>(matrix_types::upper, n, n, my_rand));

	matrix_columns<double> b = generate_matrix<matrix_columns<type>>(matrix_types::full, n, m, my_rand);

	ofstream to_in(string("input") + file_descr, ios::out | ios::binary);
	to_in.write(reinterpret_cast<char*>(&base_seed), sizeof(base_seed));
	write_matrix_to_binary_file(to_in, a);
	write_matrix_to_binary_file(to_in, b);
	to_in.close();

	auto begin = chrono::high_resolution_clock::now();

	auto ans = (mt == 0 ?
		solve_lower_system_mkl<decltype(a), decltype(b), matrix_columns<double>>(a, b, n, m)
		: solve_upper_system_mkl<decltype(a), decltype(b), matrix_columns<double>>(a, b, n, m));

	auto end = chrono::high_resolution_clock::now();
	auto elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	to_in.open(string("answer") + file_descr, ios::out | ios::binary);
	write_matrix_to_binary_file(to_in, ans);
	to_in.close();


	double max_error = 0;
	for (int64_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			double s = 0;
			for (size_t k = 0; k < n; k++)
			{
				s += a[i][k] * ans[k][j];
			}

			max_error = max(max_error, abs(s - b[i][j]));
		}
	}

	if (max_error > std::numeric_limits<type>::epsilon() * 1e+10) {
		ofstream fout(string("bad_tests.log") + file_descr, ios::out | ios::app);

		if (mt == 0)
			fout << "hight error of mkl in test: lower system n = " << n << " m = " << m << " seed = " << base_seed << " err = " << max_error << endl;
		else 
			fout << "hight error of mkl in test: upper system n = " << n << " m = " << m << " seed = " << base_seed << " err = " << max_error << endl;

		fout.close();
	}

	ofstream time_report(string("mkl_times") + file_descr, ios::app);
	time_report << elapsed_secs << ' ';
	time_report.close();
}

void test(size_t n, size_t m, int test_type, const std::string& file_descr)
{
	ifstream fin(string("input") + file_descr, ios::in | ios::binary);
	
	int64_t base_seed;
	fin.read(reinterpret_cast<char*>(&base_seed), sizeof(base_seed));

	matrix<type> a(n, n);
	matrix_columns<type> b(n, m);

	read_matrix_from_binary_file(fin, a);
	read_matrix_from_binary_file(fin, b);

	fin.close();

	fin.open(string("answer") + file_descr, ios::in | ios::binary);

	matrix<double> ans(n, m);
	read_matrix_from_binary_file(fin, ans);

	fin.close();

	auto tested_func = solve_lower_system_blocks<matrix<double>, matrix_columns<double>, matrix_columns<double>>;

	switch (test_type)
	{
	case 0:
		tested_func = solve_lower_system_blocks<matrix<double>, matrix_columns<double>, matrix_columns<double>>;
		break;
	case 1:
		tested_func = solve_lower_system_blocks_mkl_mul<matrix<double>, matrix_columns<double>, matrix_columns<double>>;
		break;
	case 2:
		tested_func = solve_upper_system_blocks<matrix<double>, matrix_columns<double>, matrix_columns<double>>;
		break;
	case 3:
		tested_func = solve_upper_system_blocks_mkl_mul<matrix<double>, matrix_columns<double>, matrix_columns<double>>;
		break;
	}

	auto res = tester(tested_func, a, b, n, m, ans);

	if (res.second > std::numeric_limits<type>::epsilon() * 1e+10) {
		ofstream fout(string("bad_tests.log") + file_descr, ios::out | ios::app);

		fout << "hight error in test: type = " << test_type << " n = " << n << " m = " << m << " seed = " << base_seed << " err = " << res.second << endl;

		fout.close();
	}

	ofstream rep(string("time_report") + file_descr, ios::app);
	rep << res.first << ' ';
	rep.close();
}

//#define CHECK

int main(int argc, char* argv[])
{

#ifndef CHECK
	size_t n = atoi(argv[1]);
	size_t thread_nums = atoi(argv[2]);
	int test_type = atoi(argv[3]); // -1 - create   0 - without mkl    1 - witch mkl
	int sys_type = atoi(argv[4]); // 0 - lower   1 - upper
	string descr = argv[5];

#else
	size_t n = 5000;
	size_t thread_nums = 1;
	int test_type = 0; // -1 - create   0 - without mkl    1 - witch mkl
	int sys_type = 0; // 0 - lower   1 - upper
	string descr = "_test";
#endif

	omp_set_num_threads(thread_nums);

	if (test_type == -1) {
		size_t seed = random_device{}();
		mt19937 my_rand(seed);

		create_test(n, n, my_rand, descr, sys_type, seed);
	}
	else {
		test(n, n, sys_type * 2 + test_type, descr);
	}

	return 0;
}
