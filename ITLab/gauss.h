#pragma once

#include <vector>
#include <utility>
#include <type_traits>
#include <mkl.h>
#include <algorithm>
#include "Matrix.h"

#ifdef FIRST_BLOCK_SIZE
constexpr size_t FIRST_GLOBAL_BLOCK_SIZE_BYTE = FIRST_BLOCK_SIZE;
#else
constexpr size_t FIRST_GLOBAL_BLOCK_SIZE_BYTE = 512;
#endif

#ifdef SECOND_BLOCK_SIZE
constexpr size_t SECOND_GLOBAL_BLOCK_SIZE_BYTE = SECOND_BLOCK_SIZE;
#else
constexpr size_t SECOND_GLOBAL_BLOCK_SIZE_BYTE = 512;
#endif

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


template <typename T>
matrix_columns<T> solve_lower_system_blocks(const matrix<T>& a, const matrix_columns<T>& b, size_t n, size_t m)
{
	using value_type = T;
	constexpr size_t first_block_size = FIRST_GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);
	constexpr size_t second_block_size = SECOND_GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	matrix_columns<T> ans(n, m, noinit);

	for (size_t i = 0; i < n; i++)
	{
	#pragma omp parallel for schedule(static)
		for (size_t second_block = 0; second_block < m; second_block += second_block_size)
		{
			for (size_t j = second_block; j < m && j < second_block + second_block_size; j++)
			{
				ans[i][j] = 0;
			}
		}
	}

	for (size_t first_block = 0; first_block < n; first_block += first_block_size)
	{
	#pragma omp parallel for schedule(static)
		for (size_t second_block = 0; second_block < m; second_block += second_block_size)
		{
			size_t er = std::min<size_t>(second_block + second_block_size, m);
			for (size_t i = first_block; i < n && i < first_block + first_block_size; i++)
			{
				for (int64_t j = second_block; j < er; j++)
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


			for (size_t nfirst_block = first_block + first_block_size; nfirst_block < n; nfirst_block += first_block_size)
			{
				size_t el = std::min<size_t>(nfirst_block + first_block_size, n);
				for (size_t i = nfirst_block; i < el; i++)
				{
					for (size_t j = second_block; j < er; j++)
					{
						auto sum = ans[i][j];
					#pragma omp simd reduction(+: sum)
						for (size_t k = first_block; k < first_block + first_block_size; k++)
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

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = FIRST_GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	R_T ans(a_size, b_columns);

	for (size_t i = 0; i < a_size; i++)
	{
	#pragma omp parallel for schedule(static)
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			for (size_t j = 0; j < second_block && j < b_columns; j++)
			{
				ans[i][j] = 0;
			}
		}
	}

	for (size_t first_block = 0; first_block < a_size; first_block += block_size)
	{
	#pragma omp parallel for schedule(static)
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

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = FIRST_GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	R_T ans(a_size, b_columns);

	for (size_t i = 0; i < a_size; i++)
	{
	#pragma omp parallel for schedule(static)
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			for (size_t j = 0; j < second_block && j < b_columns; j++)
			{
				ans[i][j] = 0;
			}
		}
	}

	for (size_t first_block = a_size - a_size % block_size; first_block + block_size > 0; first_block -= block_size)
	{
		size_t block = std::min(a_size - first_block, block_size);
	#pragma omp parallel for schedule(static)
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
	#pragma omp parallel for collapse(2) schedule(static)
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

	using value_type = typename std::decay<decltype(a[0][0])>::type;

	constexpr size_t block_size = FIRST_GLOBAL_BLOCK_SIZE_BYTE / sizeof(value_type);

	R_T ans(a_size, b_columns);

	for (size_t first_block = 0; first_block < a_size; first_block += block_size)
	{
	#pragma omp parallel for schedule(static)
		for (size_t second_block = 0; second_block < b_columns; second_block += block_size)
		{
			for (size_t i = first_block; i < a_size && i < first_block + block_size; i++)
			{
				size_t e = std::min(b_columns, second_block + block_size);
				for (int64_t j = second_block; j < e; j++)
				{
					ans[i][j] = 0;
				}
			}
		}
	}

	for (size_t first_block = a_size - a_size % block_size; first_block + block_size > 0; first_block -= block_size)
	{
		size_t block = std::min(a_size - first_block, block_size);
	#pragma omp parallel for schedule(static)
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


template <typename T>
matrix_columns<T> solve_lower_system_mkl(const matrix<T>& a, const matrix_columns<T>& b, size_t a_size, size_t b_columns)
{
	matrix_columns<T> ans(a_size, b_columns, noinit);

#pragma omp parallel for
	for (int64_t j = 0; j < b_columns; j++)
	{
		for (size_t i = 0; i < a_size; i++)
		{
			ans[i][j] = b[i][j];
		}
	}

	cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, a_size, b_columns, 1, &a[0][0], a_size, &ans[0][0], a_size);

	return ans;
}


template <typename L_M, typename M, typename R_T>
R_T solve_upper_system_mkl(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	//static_assert(false);


	R_T ans(a_size, b_columns);

	for (size_t i = 0; i < a_size; i++)
	{
	#pragma omp parallel for
		for (int64_t j = 0; j < b_columns; j++)
		{
			ans[i][j] = b[i][j];
		}
	}

	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, a_size, b_columns, 1, &a[0][0], a_size, &ans[0][0], a_size);

	return ans;
}
