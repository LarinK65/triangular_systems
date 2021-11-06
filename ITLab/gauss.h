#pragma once

#include <vector>
#include <utility>

template <typename L_M, typename M, typename R_T>
R_T solve_lower_system_basic(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(a_size, b_columns);

	for (int64_t i = 0; i < a_size; i++)
	{
	#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			auto sum = ans[i][j];
		#pragma omp simd reduction(+: sum)
			for (size_t k = 0; k < i; k++)
			{
				sum += a[i][k] * ans[k][j];
			}

			ans[i][j] = b[i][j] - sum;
			ans[i][j] /= a[i][i];
		}
	}


	return ans;
}



template <typename L_M, typename M, typename R_T>
R_T solve_lower_system_blocks(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(a_size, b_columns);

	using value_type = std::decay_t<decltype(a[0][0])>;

	constexpr size_t block_size = 512 / sizeof(value_type);

	for (size_t first_block = 0; first_block < a_size; first_block += block_size)
	{
	#pragma omp parallel for
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			for (size_t i = first_block; i < a_size && i < first_block + block_size; i++)
			{
				size_t e = std::min(b_columns, second_block + block_size);
				for (int64_t j = second_block; j < e; j++)
				{
					auto sum = ans[i][j];
				#pragma omp simd reduction(+: sum)
					for (size_t k = first_block; k < i; k++)
					{
						sum += a[i][k] * ans[k][j];
					}

					ans[i][j] = b[i][j] - sum;
					ans[i][j] /= a[i][i];
				}
			}
		}

		#pragma omp parallel for
		for (size_t nfirst_block = first_block + block_size; nfirst_block < a_size; nfirst_block += block_size)
		{
			for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
			{
				for (size_t i = nfirst_block; i < a_size && i < nfirst_block + block_size; i++)
				{
					size_t e = std::min(b_columns, second_block + block_size);

					for (size_t j = second_block; j < e; j++)
					{
						auto sum = ans[i][j];
					#pragma omp simd reduction(+: sum)
						for (size_t k = first_block; k < first_block + block_size; k++)
						{
							sum += a[i][k] * ans[k][j];
						}

						ans[i][j] = sum;
					}
				}
			}
		}
	}

	return ans;
}

