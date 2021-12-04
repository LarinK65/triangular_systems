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

void create_test(size_t n, size_t m, std::mt19937& my_rand, int64_t base_seed)
{
	matrix<type> a = generate_matrix<matrix<type>>(matrix_types::lower, n, n, my_rand);

	matrix_columns<type> b = generate_matrix<matrix_columns<type>>(matrix_types::full, n, m, my_rand);

	ofstream to_in("input", ios::out | ios::binary);
	write_matrix_to_binary_file(to_in, a);
	write_matrix_to_binary_file(to_in, b);
	to_in.close();

	auto ans = solve_lower_system_mkl<type>(a, b, n, m);

	to_in.open("answer", ios::out | ios::binary);
	write_matrix_to_binary_file(to_in, ans);
	to_in.close();

	cout << base_seed << '\n' << endl;

	double max_error = 0;
	for (size_t i = 0; i < n; i++)
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

	cout << "mkl error is: " << max_error << '\n';
}

void test(size_t n, size_t m, int test_type, const std::string& input_file, const std::string& ans_file, const std::string& test_handle)
{
	ifstream fin(input_file, ios::in | ios::binary);

	if (!fin.is_open()) {
		cerr << "cant open file\n";
		exit(-1);
	}

	matrix<type> a(n, n, noinit);
	matrix_columns<type> b(n, m, noinit);

	read_matrix_from_binary_file(fin, a);
	read_matrix_from_binary_file(fin, b);

	fin.close();

	
	
	matrix_columns<type> ans(n, m, noinit);

	fin.open(ans_file, ios::in | ios::binary);

	read_matrix_from_binary_file(fin, ans);

	fin.close();

	auto tested_func = solve_lower_system_blocks<type>;

	if (test_type == 0) {
		tested_func = solve_lower_system_blocks<type>;
	}
	else if (test_type == 1) {
		tested_func = solve_lower_system_mkl<type>;
	}


	auto res = tester(tested_func, a, b, n, m, ans);

	if (res.second > std::numeric_limits<type>::epsilon() * 1e+6) {
		ofstream fout(test_handle + string("_bad_tests.log"), ios::out | ios::app);

		fout << "hight error in test: err = " << res.second << endl;

		fout.close();
	}

	ofstream rep(test_handle + string("time_report.rep"), ios::app);
	rep << res.first << ' ';
	rep.close();
}

//#define GENERATE

int main(int argc, char* argv[])
{

#ifndef GENERATE
	size_t n = atoi(argv[1]);
	size_t thread_nums = atoi(argv[2]);
	int test_type = atoi(argv[3]); // -1 - create   0 - my_alg    1 - mkl
	string input_file = argv[4];
	string answer_file = argv[5];
	string test_handle = argv[6];

#else
	size_t n;
	cin >> n;
	size_t thread_nums = 1;
	int test_type = -1;
	string input_file;
	string answer_file;
	string test_handle;
#endif

	omp_set_num_threads(thread_nums);

	if (test_type == -1) {
		size_t seed = random_device{}();
		mt19937 my_rand(seed);

		create_test(n, n, my_rand, seed);
	}
	else if (test_type == 0) {
		test(n, n, 0, input_file, answer_file, test_handle);
	}
	else {
		test(n, n, 1, input_file, answer_file, test_handle);
	}

	return 0;
}
