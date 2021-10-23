#pragma once

#include <vector>
#include <utility>


template <typename U_M, typename M>
M calc_U(const U_M& a, const M& b, size_t a_size, size_t b_columns)
{
	M ans(a_size, b_columns);

	for (int64_t i = a_size - 1; i + 1 > 0; i--)
	{
#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			for (size_t k = i + 1; k < a_size; k++)
			{
				ans[i][j] += a[i][k] * ans[k][j];
			}

			ans[i][j] = b[i][j] - ans[i][j];
			
			ans[i][j] /= a[i][i];
		}
	}


	return ans;
}



template <typename L_M, typename M>
M calc_L(const L_M& a, const M& b,  size_t a_size, size_t b_columns)
{
	M ans(a_size, b_columns);

	for (int64_t i = 0; i < a_size; i++)
	{
	#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			for (size_t k = 0; k < i; k++)
			{
				ans[i][j] += a[i][k] * ans[k][j];
			}

			ans[i][j] = b[i][j] - ans[i][j];

			ans[i][j] /= a[i][i];
		}
	}


	return ans;
}


template <typename L_M, typename M>
M solve_lower_system_blocks(const L_M& a, const M& b,  size_t a_size, size_t b_columns)
{
	M ans(a_size, b_columns);

	using value_type = std::decay_t<decltype(b[0][0])>;

	constexpr size_t block_size = 2048 / sizeof(value_type);

	for (size_t i = 0; i < a_size; i++)
	{
		for (int64_t start_block = 0; start_block < i; start_block += block_size)
		{
		#pragma omp parallel for
			for (int64_t j = 0; j < b_columns; j++)
			{
				value_type sum = 0;
				for (int64_t k = start_block; k < i && k < start_block + block_size; k++)
				{
					sum += a[i][k] * ans[k][j];
				}

				ans[i][j] += sum;
			}
		}


		#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			ans[i][j] = b[i][j] - ans[i][j];
			ans[i][j] /= a[i][i];
		}
	}

	return ans;
}

template <typename U_M, typename M>
M solve_upper_system_blocks(const U_M& a, const M& b,  size_t a_size, size_t b_columns)
{
	M ans(a_size, b_columns);

	using value_type = std::decay_t<decltype(b[0][0])>;

	constexpr size_t block_size = 2048 / sizeof(value_type);

	for (size_t i = a_size - 1; i + 1 > 0; i--)
	{
		for (int64_t start_block = i + 1; start_block < a_size; start_block += block_size)
		{
		#pragma omp parallel for
			for (int64_t j = 0; j < b_columns; j++)
			{
				value_type sum = 0;
				for (int64_t k = start_block; k < a_size && k < start_block + block_size; k++)
				{
					sum += a[i][k] * ans[k][j];
				}

				ans[i][j] += sum;
			}
		}


	#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			ans[i][j] = b[i][j] - ans[i][j];
			ans[i][j] /= a[i][i];
		}
	}

	return ans;
}
