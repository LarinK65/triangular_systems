#pragma once
#include <vector>

template <typename T, bool IsUpper>
struct triangle_matrix
{
public:
	triangle_matrix() = default;
	triangle_matrix(const triangle_matrix&) = default;
	triangle_matrix(triangle_matrix&&) = default;

	triangle_matrix& operator=(const triangle_matrix&) = default;
	triangle_matrix& operator=(triangle_matrix&&) = default;


	triangle_matrix(size_t sz)
		: data(sz* (sz + 1) / 2)
		, _size(sz)
	{}

	triangle_matrix(size_t sz, const T& value)
		: data(sz* (sz + 1) / 2, value)
		, _size(sz)
	{}

	template <typename Matrix>
	triangle_matrix(size_t sz, Matrix m)
		: data(sz* (sz + 1) / 2)
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

protected:
	std::vector<T> data;
	size_t _size;
};

template <typename T, bool IsUpper>
class triangle_matrix_full;

template <typename T, bool IsUpper>
struct triangle_matrix_value_reference
{
	using matrix_type =  triangle_matrix_full<T, IsUpper>;

	triangle_matrix_value_reference(matrix_type& m, size_t i, size_t j)
		: i(i)
		, j(j)
		, matrix_ref(m)
	{}

	triangle_matrix_value_reference& operator=(const T& value)
	{
		if constexpr (IsUpper) {
			if (i <= j) {
				this->matrix_ref.data[(2 * this->matrix_ref->_size - i - 1) * i / 2 + j] = value;
			}
		}
		else {
			if (i >= j) {
				this->matrix_ref.data[i * (i + 1) / 2 + j] = value;
			}
		}

		return *this;
	}
	triangle_matrix_value_reference& operator=(const triangle_matrix_value_reference& right)
	{
		*this = static_cast<T>(right);
		return *this;
	}

	operator T() const noexcept
	{
		if constexpr (IsUpper) {
			if (i > j) {
				return 0;
			}
			else {
				return this->matrix_ref.data[(2 * this->matrix_ref._size - i - 1) * i / 2 + j];
			}
		}
		else {
			if (i < j) {
				return 0;
			}
			else {
				return this->matrix_ref.data[i * (i + 1) / 2 + j];
			}
		}
	}

	size_t i, j;
	matrix_type& matrix_ref;
};

template <typename T, bool IsUpper>
struct row_reference_triangle_matrix_full
{
	using matrix_type =  triangle_matrix_full<T, IsUpper>;
	using reference = triangle_matrix_value_reference<T, IsUpper>;

	row_reference_triangle_matrix_full(const matrix_type& m, size_t i)
		: i(i)
		, matrix_ref(m)
	{}

	reference operator[](size_t j)
	{
		return reference(const_cast<matrix_type&>(matrix_ref), i, j);
	}
	const T& operator[](size_t j) const
	{
		if constexpr (IsUpper) {
			if (i > j) {
				return 0;
			}
			else {
				return this->matrix_ref.data[(2 * this->matrix_ref._size - i - 1) * i / 2 + j];
			}
		}
		else {
			if (i < j) {
				return 0;
			}
			else {
				return this->matrix_ref.data[i * (i + 1) / 2 + j];
			}
		}
	}

	size_t i;
	const matrix_type& matrix_ref;
};

template <typename T, bool IsUpper>
class triangle_matrix_full : triangle_matrix<T, IsUpper>
{
public:
	using triangle_matrix<T, IsUpper>::triangle_matrix;
	triangle_matrix_full(const triangle_matrix<T, IsUpper>& other) : triangle_matrix<T, IsUpper>(other) {}

	row_reference_triangle_matrix_full<T, IsUpper> operator[](size_t i)
	{
		return row_reference_triangle_matrix_full<T, IsUpper>(*this, i);
	}
	const row_reference_triangle_matrix_full<T, IsUpper> operator[](size_t i) const
	{
		return const_cast<triangle_matrix_full&>(*this)[i];
	}

	friend struct row_reference_triangle_matrix_full<T, IsUpper>;
	friend struct triangle_matrix_value_reference<T, IsUpper>;
};



template <typename T>
struct matrix_columns;

template <typename T>
struct matrix
{
	matrix() = default;
	matrix(const matrix&) = default;
	matrix(matrix&&) = default;

	matrix& operator=(const matrix&) = default;
	matrix& operator=(matrix&&) = default;

	matrix(size_t rows, size_t columns)
		: h(rows)
		, w(columns)
		, data(h* columns)
	{}

	matrix(size_t rows, size_t columns, const T& value)
		: h(rows)
		, w(columns)
		, data(rows* columns, value)
	{}

	template <typename M>
	matrix(size_t rows, size_t columns, const M& m)
		: h(rows)
		, w(columns)
		, data(rows* columns)
	{
		for (size_t i = 0; i < this->h; i++)
		{
			for (size_t j = 0; j < this->w; j++)
			{
				(*this)[i][j] = m[i][j];
			}
		}
	}

	T* operator[](size_t i) noexcept
	{
		return this->data.data() + i * this->w;
	}
	const T* operator[](size_t i) const noexcept
	{
		return const_cast<matrix&>(*this)[i];
	}

	friend bool operator==(const matrix& l, const matrix& r)
	{
		return l.data == r.data;
	}
	friend bool operator!=(const matrix& l, const matrix& r)
	{
		return !(l == r);
	}


	size_t h, w;
	std::vector<T> data;
};


template <typename T>
struct row_reference
{
	row_reference(const std::vector<T>& matrix_data, size_t i, size_t h)
		: data(matrix_data)
		, row(i)
		, h(h)
	{}

	const T& operator[](size_t j) const noexcept
	{
		return this->data[j * h + row];
	}
	T& operator[](size_t j) noexcept
	{
		return const_cast<T&>(static_cast<const row_reference&>(*this)[j]);
	}

private:
	size_t row, h;
	const std::vector<T>& data;
};

template <typename T>
struct matrix_columns : matrix<T>
{
	using matrix<T>::matrix;
	matrix_columns(const matrix<T>& other)
		: matrix<T>(other.h, other.w)
	{
		for (size_t i = 0; i < this->h; i++)
		{
			for (size_t j = 0; j < this->w; j++)
			{
				(*this)[i][j] = other[i][j];
			}
		}
	}

	row_reference<T> operator[](size_t i) noexcept
	{
		return row_reference(this->data, i, this->h);
	}
	const row_reference<T> operator[](size_t i) const noexcept
	{
		return const_cast<matrix_columns&>(*this)[i];
	}

};

