#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include <mkl.h>


constexpr size_t GLOBAL_BLOCK_SIZE_BYTE = 512;

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

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

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

	#pragma omp parallel for collapse(2)
		for (size_t nfirst_block = first_block + block_size; nfirst_block < a_size; nfirst_block += block_size)
		{
			for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
			{
				size_t e = std::min(b_columns, second_block + block_size);
				for (size_t i = nfirst_block; i < a_size && i < nfirst_block + block_size; i++)
				{

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



template <typename L_M, typename M, typename R_T>
R_T solve_lower_system_blocks_mkl_mul(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(a_size, b_columns);

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	for (size_t first_block = 0; first_block < a_size; first_block += block_size)
	{
	#pragma omp parallel for
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			size_t e = std::min(b_columns, second_block + block_size);
			for (size_t i = first_block; i < a_size && i < first_block + block_size; i++)
			{
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


		if (a_size > first_block + block_size) {
			cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a_size - first_block - block_size, b_columns, min(block_size, a_size - first_block), 1,
				&a[first_block + block_size][first_block], a_size, &ans[first_block][0], a_size, 1, &ans[first_block + block_size][0], a_size);
		}
	}

	return ans;
}


template <typename L_M, typename M, typename R_T>
R_T solve_upper_system_blocks(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(a_size, b_columns);

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	for (size_t first_block = a_size - a_size % block_size; first_block + block_size > 0; first_block -= block_size)
	{
		size_t block = std::min(a_size - first_block, block_size);
	#pragma omp parallel for
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			size_t e = std::min(b_columns, second_block + block_size);
			for (size_t i = first_block + block - 1; i + 1 > first_block; i--)
			{
				for (size_t j = second_block; j < e; j++)
				{
					auto sum = ans[i][j];
				#pragma omp simd reduction(+: sum)
					for (size_t k = i + 1; k < first_block + block; k++)
					{
						sum += a[i][k] * ans[k][j];
					}

					ans[i][j] = b[i][j] - sum;
					ans[i][j] /= a[i][i];
				}
			}
		}

		size_t e2 = std::min(a_size, first_block + block_size);
	#pragma omp parallel for collapse(2)
		for (size_t nfirst_block = 0; nfirst_block < first_block; nfirst_block += block_size)
		{
			for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
			{
				for (size_t i = nfirst_block; i < nfirst_block + block_size && i < a_size; i++)
				{
					size_t e1 = std::min(b_columns, second_block + block_size);

					for (size_t j = second_block; j < e1; j++)
					{
						auto sum = ans[i][j];
					#pragma omp simd reduction(+: sum)
						for (size_t k = first_block; k < e2; k++)
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



template <typename L_M, typename M, typename R_T>
R_T solve_upper_system_blocks_mkl_mul(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(a_size, b_columns);

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	for (size_t first_block = a_size - a_size % block_size; first_block + block_size > 0; first_block -= block_size)
	{
		size_t block = std::min(a_size - first_block, block_size);
	#pragma omp parallel for
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			size_t e = std::min(b_columns, second_block + block_size);
			for (size_t i = first_block + block - 1; i + 1 > first_block; i--)
			{
				for (size_t j = second_block; j < e; j++)
				{
					auto sum = ans[i][j];
				#pragma omp simd reduction(+: sum)
					for (size_t k = i + 1; k < first_block + block; k++)
					{
						sum += a[i][k] * ans[k][j];
					}

					ans[i][j] = b[i][j] - sum;
					ans[i][j] /= a[i][i];
				}
			}
		}


		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, first_block, b_columns, block, 1,
			&a[0][first_block], a_size, &ans[first_block][0], a_size, 1, &ans[0][0], a_size);
	}

	return ans;
}


template <typename L_M, typename M, typename R_T>
R_T solve_lower_system_mkl(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(b);

	cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, a_size, b_columns, 1, &a[0][0], a_size, &ans[0][0], a_size);

	return ans;
}


template <typename L_M, typename M, typename R_T>
R_T solve_upper_system_mkl(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	R_T ans(b);

	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, a_size, b_columns, 1, &a[0][0], a_size, &ans[0][0], a_size);

	return ans;
}
