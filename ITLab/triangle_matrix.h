#pragma once
#include <vector>

template <typename T, bool IsUpper>
class triangle_matrix
{
public:
	triangle_matrix() = default;
	triangle_matrix(const triangle_matrix&) = default;
	triangle_matrix(triangle_matrix&&) = default;

	triangle_matrix& operator=(const triangle_matrix&) = default;
	triangle_matrix& operator=(triangle_matrix&&) = default;


	triangle_matrix(size_t sz)
		: data(sz * (sz + 1) / 2)
		, _size(sz)
	{}

	triangle_matrix(size_t sz, const T& value)
		: data(sz * (sz + 1) / 2, value)
		, _size(sz)
	{}

	template <typename Matrix>
	triangle_matrix(size_t sz, Matrix m)
		: data(sz * (sz + 1) / 2)
		, _size(sz)
	{
		for (size_t i = 0; i < sz; i++)
		{
			if constexpr (IsUpper) {
				for (size_t j = i; j < sz; j++)
				{
					(*this)[i][j] = m[i][j];
				}
			}
			else {
				for (size_t j = 0; j <= i; j++)
				{
					(*this)[i][j] = m[i][j];
				}
			}
		}
	}

	T* operator[](size_t row) noexcept
	{
		if constexpr (IsUpper) {
			return this->data.data() + (2 * this->_size - row - 1) * row / 2;
		}
		else {
			return this->data.data() + (row + 1) * row / 2;
		}
	}
	const T* operator[](size_t row) const noexcept
	{
		return const_cast<triangle_matrix&>(*this)[row];
	}

	T& at(size_t i, size_t j) noexcept
	{
		if constexpr (IsUpper) {
			return this->data[(2 * this->_size - i - 1) * i / 2 + j];
		}
		return this->data[i * (i + 1) / 2 + j];
	}

	const T& at(size_t i, size_t j) const noexcept
	{
		return const_cast<triangle_matrix&>(*this).at(i, j);
	}

	friend bool operator==(const triangle_matrix& l, const triangle_matrix& r) noexcept
	{
		return l.data == r.data;
	}

private:
	std::vector<T> data;
	size_t _size;
};

