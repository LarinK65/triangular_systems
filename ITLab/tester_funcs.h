#pragma once

#include "Matrix.h"
#include <random>
#include <utility>
#include <chrono>

enum class matrix_types {full, upper, lower};

std::uniform_real_distribution<float> distr(-1'000, 1'000);
std::uniform_int_distribution<int> mid_distr(10'000, 100'000);

template <typename MT>
MT generate_matrix(matrix_types gen_type, size_t n, size_t m, std::mt19937& my_rand)
{
	MT res(n, m);

	switch (gen_type)
	{
	case matrix_types::full:
		for (size_t i = 0; i < n; i++)
		{
			for (size_t j = 0; j < m; j++)
			{
				res[i][j] = distr(my_rand);
			}
		}

		break;

	case matrix_types::upper:
		for (int64_t i = 0; i < n; i++)
		{
			for (size_t j = i + 1; j < m; j++)
			{
				res[i][j] = distr(my_rand);
			}
			res[i][i] = mid_distr(my_rand);
			if (mid_distr(my_rand) % 2) {
				res[i][i] = -res[i][i];
			}
		}

		break;

	case matrix_types::lower:
		for (int64_t i = 0; i < n; i++)
		{
			for (size_t j = 0; j < i; j++)
			{
				res[i][j] = distr(my_rand);
			}
			res[i][i] = mid_distr(my_rand);
			if (mid_distr(my_rand) % 2) {
				res[i][i] = -res[i][i];
			}
		}

		break;
	}

	return res;
}


template <typename func, typename T>
std::pair<int64_t, double> tester(func tested_function, const matrix<T>& a, const matrix_columns<T>& b, size_t n, size_t m, const matrix_columns<T>& ans)
{
	auto begin = std::chrono::high_resolution_clock::now();
	auto res = tested_function(a, b, n, m);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	double max_err = 0;
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			max_err = max(max_err, res[i][j] - ans[i][j]);
		}
	}

	return { elapsed_secs, max_err };
}

template <typename MT>
void read_matrix_from_binary_file(std::istream& in, MT& m)
{
	for (size_t i = 0; i < m.h; i++)
	{
		for (size_t j = 0; j < m.w; j++)
		{
			in.read(reinterpret_cast<char*>(&m[i][j]), sizeof(m[i][j]));
		}
	}
}

template <typename MT>
void write_matrix_to_binary_file(std::ostream& out, const MT& m)
{
	for (size_t i = 0; i < m.h; i++)
	{
		for (size_t j = 0; j < m.w; j++)
		{
			out.write(reinterpret_cast<const char*>(&m[i][j]), sizeof(m[i][j]));
		}
	}
}


