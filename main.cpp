#include <iostream>
#include <utility>
#include <vector>
#include <iomanip>
#include <valarray>
#include <random>

using namespace std;
struct DimensionException{
    DimensionException(int n_a, int m_a, int n_b, int m_b) {
        this->n_a = n_a;
        this->n_b = n_b;
        this->m_a = m_a;
        this->m_b = m_b;
    }

    int n_a;
    int n_b;
    int m_a;
    int m_b;
};
struct ApplicableException {

};
class Matrix {
protected:

    int n;
    int m;
    vector<vector<double>> matrix;
    static void roundVal(double &val) {
        if (abs(val) < 0.0000000001) {
            val = 0;
        }
    }

    static double column_multiplication(vector<double> a, vector<double> b) {
        int size = a.size();
        double product = 0;
        for (int i = 0; i < size; i++) {
            product += a[i]* b[i];
        }
        return product;
    }

public:
    Matrix(int mx_n, int mx_m, vector<vector<double>> mx_values) {
        m = mx_m;
        n = mx_n;
        matrix = std::move(mx_values);
    }
    Matrix() {
        m = 0;
        n = 0;
        matrix = vector<vector<double>>();
    }
    Matrix(int mx_n, int mx_m) {
        m = mx_m;
        n = mx_n;
        matrix = vector<vector<double>>();
        for (int i = 0; i < n; i++) {
            matrix.push_back(*new vector<double>(m));
        }
    }
    Matrix(Matrix& mx, int start_row, int end_row, int start_column, int end_column) :
            Matrix(end_row - start_row, end_column - start_column) {
        double val;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                val = mx[i+start_row][j+start_column];
                matrix[i][j] = val;
            }
        }

    }

    friend ostream& operator<<(ostream& output, const Matrix& mx) {
        for (int i = 0; i < mx.n; i++) {
            output << fixed << setprecision(4) << mx.matrix[i][0];
            for (int j = 1; j < mx.m; j++) {
                output << " "<< fixed << setprecision(4) << mx.matrix[i][j];
            }
            output << endl;
        }
        return output;
    }
    friend istream& operator>>(istream& input, Matrix& mx) {
        for (int i = 0; i < mx.n; i++) {
            for (int j = 0; j < mx.m; j++) {
                input >> mx.matrix[i][j];
            }
        }
        return input;
    }
    Matrix& operator=(const Matrix& mx) = default;

    Matrix* operator+(Matrix& mx) {
        if ((n!=mx.n) || (m!= mx.m)) {
            throw DimensionException(n, m, mx.n, mx.m);
        } else {
            vector<vector<double>> mx_value = vector<vector<double>>();
            for (int i = 0; i < n; i++) {
                mx_value.push_back(*new vector<double>(m));
                for (int j = 0; j < m; j++) {
                    mx_value[i][j] = matrix[i][j] + mx.matrix[i][j];
                    roundVal(mx_value[i][j]);
                }
            }
            Matrix* res = new Matrix(n, m, mx_value);
            return res;
        }
    }

    Matrix* operator-(Matrix& mx) {
        if ((n!=mx.n) || (m!= mx.m)) {
            throw DimensionException(n, m, mx.n, mx.m);
        } else {
            vector<vector<double>> mx_value = vector<vector<double>>();
            for (int i = 0; i < n; i++) {
                mx_value.push_back(*new vector<double>(m));
                for (int j = 0; j < m; j++) {
                    mx_value[i][j] = matrix[i][j] - mx.matrix[i][j];
                    roundVal(mx_value[i][j]);
                }
            }
            Matrix* res = new Matrix(n, m, mx_value);
            return res;
        }
    }
    vector<double>& operator[](int i) {
        return matrix[i];
    }
    Matrix* transpose() {
        vector<vector<double>> mx_value = vector<vector<double>>();
        for (int i = 0; i < m; i++) {
            mx_value.push_back(*new vector<double>(n));
            for (int j = 0; j < n; j++) {
                mx_value[i][j] = matrix[j][i];
            }
        }
        Matrix* res = new Matrix(m, n, mx_value);
        return res;
    }

    Matrix* operator*(Matrix mx) {
        if (m != mx.n) {
            throw DimensionException(n, m, mx.n, mx.m);
        } else {
            Matrix* mx_transpose = mx.transpose();
            vector<vector<double>> mx_value = vector<vector<double>>();
            for (int i = 0; i < n; i++) {
                mx_value.push_back(*new vector<double>(mx.m));
                for (int j = 0; j < mx.m; j++) {
                    mx_value[i][j] = column_multiplication(matrix[i], mx_transpose->matrix[j]);
                    roundVal(mx_value[i][j]);
                }
            }
            Matrix* res = new Matrix(n, mx.m, mx_value);
            return res;
        }
    }
    Matrix& operator*=(Matrix mx) {
        if (m != mx.n) {
            throw DimensionException(n, m, mx.n, mx.m);
        } else {
            Matrix* mx_transpose = mx.transpose();
            vector<vector<double>> mx_value = vector<vector<double>>();
            for (int i = 0; i < n; i++) {
                mx_value.push_back(*new vector<double>(mx.m));
                for (int j = 0; j < mx.m; j++) {
                    mx_value[i][j] = column_multiplication(matrix[i], mx_transpose->matrix[j]);
                    roundVal(mx_value[i][j]);
                }
            }
            m = mx.m;
            matrix = mx_value;
            return *this;
        }
    }
    int get_column_number() {
        return m;
    }
    int get_row_number() {
        return n;
    }
    double compute_norm(int column) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += matrix[i][column] * matrix[i][column];
        }
        return sqrt(sum);
    }
};
class SquareMatrix: public Matrix {
public:
    SquareMatrix() : Matrix() {}
    SquareMatrix(int mx_n, vector<vector<double>> mx_values) : Matrix(mx_n, mx_n, mx_values){}
    SquareMatrix(int mx_n) : Matrix(mx_n, mx_n){}
};
class IdentityMatrix : public SquareMatrix {
public:
    IdentityMatrix(int mx_n) : SquareMatrix(mx_n) {
        for (int i = 0; i < mx_n; i++) {
            matrix[i][i] = 1;
        }
    }
};
class PermutationMatrix : public IdentityMatrix {
public:
    PermutationMatrix(int mx_n, int i_row, int j_row) : IdentityMatrix(mx_n){
        matrix[i_row][i_row] = 0;
        matrix[j_row][j_row] = 0;
        matrix[i_row][j_row] = 1;
        matrix[j_row][i_row] = 1;
    }
    PermutationMatrix(Matrix& mx, int i_row, int j_row) : IdentityMatrix(mx.get_row_number()){
        matrix[i_row][i_row] = 0;
        matrix[j_row][j_row] = 0;
        matrix[i_row][j_row] = 1;
        matrix[j_row][i_row] = 1;
    }
};
class EliminationMatrix : public IdentityMatrix {
public:
    EliminationMatrix(int mx_n, int row, int pivot, double scale) : IdentityMatrix(mx_n) {
        matrix[row][pivot] = -scale;
    }
    EliminationMatrix(Matrix& mx, int row, int pivot) : IdentityMatrix(mx.get_row_number()) {
        double scale = mx[row][pivot] / mx[pivot][pivot];
        matrix[row][pivot] = -scale;
    }
};
class DiagonalNormalizationMatrix : public IdentityMatrix {
public:
    DiagonalNormalizationMatrix(vector<double> factors) : IdentityMatrix(factors.size()) {
        int n = factors.size();
        for (int i = 0; i < n; i++) {
            matrix[i][i] = 1 / factors[i];
        }
    }
    DiagonalNormalizationMatrix(Matrix mx) : IdentityMatrix(mx.get_row_number()) {
        int n = mx.get_row_number();
        int m = mx.get_column_number();
        int bound = n < m ? n : m;
        for (int i = 0; i < bound; i++) {
            matrix[i][i] = 1 / mx[i][i];
        }
    }
};
class AugmentationMatrix : public Matrix {
protected:
    int matrix_n;
    int matrix_m;
    int augmentation_n;
    int augmentation_m;
public:
    AugmentationMatrix(Matrix& mx, Matrix& augmentation) :
            Matrix(mx.get_row_number(),mx.get_column_number() + augmentation.get_column_number()) {
        matrix_n = mx.get_row_number();
        n = matrix_n;
        matrix_m = mx.get_column_number();
        augmentation_n = augmentation.get_row_number();
        augmentation_m = augmentation.get_column_number();
        m = matrix_m + augmentation_m;
        if (matrix_n != augmentation_n) {
            throw DimensionException(matrix_n, matrix_m, augmentation_n, augmentation_m);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < matrix_m; j++) {
                matrix[i][j] = mx[i][j];
            }
            for (int j = 0; j < augmentation_m; j++) {
                matrix[i][j+matrix_m] = augmentation[i][j];
            }
        }
    }
    AugmentationMatrix(int mx_n, int mx_m, int aug_n, int aug_m, vector<vector<double>> mx_values) :
            Matrix(mx_n, mx_m+aug_m, std::move(mx_values)) {
        matrix_n = mx_n;
        matrix_m = mx_m;
        augmentation_n = aug_n;
        augmentation_m = aug_m;
        if (matrix_n != augmentation_n) {
            throw DimensionException(matrix_n, matrix_m, augmentation_n, augmentation_m);
        }
    }
    Matrix* get_matrix() {
        Matrix* mx = new Matrix(matrix_n, matrix_m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < matrix_m; j++) {
                (*mx)[i][j] = matrix[i][j];
            }
        }
        return mx;
    }
    Matrix* get_augmentation() {
        Matrix* aug = new Matrix(augmentation_n, augmentation_m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < augmentation_m; j++) {
                (*aug)[i][j] = matrix[i][j+matrix_m];
            }
        }
        return aug;
    }
};

class MatrixOperations {
protected:
    static bool pivoting(Matrix& mx, int pivot, int& step) {
        double max_val = abs(mx[pivot][pivot]);
        double val;
        int max_pos = pivot;
        int n = mx.get_row_number();
        for (int i = pivot + 1; i <n; i++) {
            val = abs(mx[i][pivot]);
            if (val > max_val) {
                max_val = val;
                max_pos = i;
            }
        }
        if (max_pos != pivot) {
            Matrix permutation = PermutationMatrix(mx, pivot, max_pos);
            mx = *(permutation * mx);
            return true;
        }
        return false;
    }
    static void elimination_step(Matrix& mx, int row, int pivot, int& step) {
        if ((mx[pivot][pivot] != 0) && (mx[row][pivot] != 0)) {
            Matrix elimination = EliminationMatrix(mx, row, pivot);
            mx = *(elimination * mx);
        }
    }
    static double getDeterminant(SquareMatrix& mx, bool reverse_sign) {
        double determinant = reverse_sign ? -1 : 1;
        int n = mx.get_row_number();
        for (int i = 0; i < n; i++) {
            determinant *= mx[i][i];
        }
        return determinant;
    }
    static void forward_elimination(Matrix& mx, int& step, bool& reverse_sign) {
        int n = mx.get_row_number();
        for (int pivot = 0; pivot < n - 1; pivot++) {
            if (pivoting(mx, pivot, step)) {
                reverse_sign = !reverse_sign;
            }
            for (int row = pivot + 1; row < n; row++) {
                elimination_step(mx, row, pivot, step);
            }
        }
    }
    static void backward_elimination(Matrix& mx, int& step) {
        int n = mx.get_row_number();
        for (int pivot = n - 1; pivot > 0; pivot--) {
            for (int row = pivot - 1; row >= 0; row--) {
                elimination_step(mx, row, pivot, step);
            }
        }
    }
    static void diagonal_normalization(Matrix& mx) {
        Matrix normalize = DiagonalNormalizationMatrix(mx);
        mx = *(normalize * mx);
    }
public:
    static double find_determinant(SquareMatrix mx) {
        bool reverse_sign = false;
        int step = 0;
        forward_elimination(mx, step, reverse_sign);
        double det = getDeterminant(mx, reverse_sign);
        return det;
    }
    static SquareMatrix* find_inverse_matrix(SquareMatrix mx) {
        int n = mx.get_row_number();
        bool inverse_sign;
        int step = 0;
        Matrix identity = IdentityMatrix(n);
        AugmentationMatrix* aug_mx = new AugmentationMatrix(mx, identity);
        forward_elimination(*aug_mx, step, inverse_sign);
        backward_elimination(*aug_mx, step);
        diagonal_normalization(*aug_mx);
        SquareMatrix* inverse = static_cast<SquareMatrix *>(aug_mx->get_augmentation());
        return inverse;
    }
};
class LeastSquareApproximation {
protected:
    static Matrix* make_equation_matrix(int data_size, Matrix& data, int degree) {
        Matrix* a = new Matrix(data_size, degree);
        double x;
        for (int m = 0; m < data_size; m++) {
            x = 1;
            for (int n = 0; n < degree; n++) {
                (*a)[m][n] = x;
                x *= data[m][0];
            }
        }
        return a;
    }
public:
    static Matrix* doApproximation(int data_size, Matrix& data, int degree) {
        degree++;
        Matrix* a = make_equation_matrix(data_size, data, degree);
        Matrix* b = new Matrix(data, 0, data_size, 1, 2);
        cout << "A:" << endl << *a;
        Matrix* aT = a->transpose();
        SquareMatrix* aTa = static_cast<SquareMatrix*>((*aT) * (*a));
        cout << "A_T*A:" << endl << *aTa;
        SquareMatrix* aTa_inv = MatrixOperations::find_inverse_matrix(*aTa);
        cout << "(A_T*A)^-1:" << endl << *aTa_inv;
        Matrix* aTb = (*aT) * (*b);
        cout << "A_T*b:" << endl << *aTb;
        Matrix* solution = (*aTa_inv) * (*aTb);
        cout << "x~:" << endl << *solution;
        return solution;
    }
};
class IterativeMethods {
protected:
    static SquareMatrix* get_lower_triangular_part(SquareMatrix& a) {
        SquareMatrix* lower_part = new SquareMatrix(a.get_row_number());
        for (int i = 0; i < a.get_row_number(); i++) {
            for (int j = 0; j < i; j++) {
                double num = a[i][j];
                (*lower_part)[i][j] = num;
            }
        }
        return lower_part;
    }
    static SquareMatrix* get_diagonal_part(SquareMatrix& a) {
        SquareMatrix* diagonal = new SquareMatrix(a.get_row_number());
        for (int i = 0; i < a.get_row_number(); i++) {
            double num = a[i][i];
            (*diagonal)[i][i] = num;
        }
        return diagonal;
    }
    static SquareMatrix* get_upper_triangular_part(SquareMatrix& a) {
        SquareMatrix* upper_part = new SquareMatrix(a.get_row_number());
        for (int i = 0; i < a.get_row_number(); i++) {
            for (int j = i+1; j < a.get_row_number(); j++) {
                double num = a[i][j];
                (*upper_part)[i][j] = num;
            }
        }
        return upper_part;
    }
    static bool check_diagonal_dominance(Matrix& a) {
        for (int i = 0; i < a.get_row_number(); i++) {
            double sum = 0;
            for (int j = 0; j < a.get_column_number(); j++) {
                sum += a[i][j];
            }
            sum -= a[i][i];
            if (abs(sum) > abs(a[i][i])) {
                return false;
            }
        }
        return true;
    }
public:
    static Matrix Jacobi_method(SquareMatrix& a, Matrix& b, double eps) {
        if (!check_diagonal_dominance(a)) {
            throw ApplicableException();
        }
        SquareMatrix* d_inv = new DiagonalNormalizationMatrix(a);
        SquareMatrix* identity = new IdentityMatrix(a.get_row_number());
        SquareMatrix* alpha = static_cast<SquareMatrix*>((*identity) - (*((*d_inv) * a)));
        cout << "alpha:" << endl << *alpha;
        Matrix* beta = (*d_inv) * b;
        cout << "beta:" << endl << *beta;
        Matrix x_prev = *beta;
        Matrix x_cur;
        int step = 0;
        double current_eps = eps+1;
        cout << "x(0):" << endl << x_prev;
        while (current_eps > eps) {
            x_cur = *(x_prev + *((*d_inv) * *(b - *(a * x_prev))));
            current_eps = (x_cur - x_prev)->compute_norm(0);
            cout << "e: " << fixed << setprecision(4) << current_eps << endl;
            cout << "x(" << ++step << "):" << endl << x_cur;
            x_prev = x_cur;
        }
        return x_cur;
    }
    static Matrix Seidel_method(SquareMatrix& a, Matrix& b, double eps) {
        if (!check_diagonal_dominance(a)) {
            throw ApplicableException();
        }
        SquareMatrix* d_inv = new DiagonalNormalizationMatrix(a);
        Matrix* beta = (*d_inv) * b;
        cout << "beta:" << endl << *beta;
        SquareMatrix* identity = new IdentityMatrix(a.get_row_number());
        SquareMatrix* alpha = static_cast<SquareMatrix*>((*identity) - (*((*d_inv) * a)));
        cout << "alpha:" << endl << *alpha;
        SquareMatrix* b_mx = get_lower_triangular_part(*alpha);
        cout << "B:" << endl << *b_mx;
        SquareMatrix* c_mx = get_upper_triangular_part(*alpha);
        cout << "C:" << endl << *c_mx;
        SquareMatrix* i_minus_b = static_cast<SquareMatrix*>(((*identity) - (*b_mx)));
        cout << "I-B:" << endl << *i_minus_b;
        SquareMatrix* i_minus_b_1 = MatrixOperations::find_inverse_matrix(*i_minus_b);
        cout << "(I-B)_-1:" << endl << *i_minus_b_1;
        Matrix x_prev = *beta;
        Matrix x_cur;
        int step = 0;
        double current_eps = eps+1;
        cout << "x(0):" << endl << x_prev;
        while (current_eps > eps) {
            x_cur = *(*(*((*i_minus_b_1) * (*c_mx)) * x_prev) + *((*i_minus_b_1) * (*beta)));
            current_eps = (x_cur - x_prev)->compute_norm(0);
            cout << "e: " << fixed << setprecision(4) << current_eps << endl;
            cout << "x(" << ++step << "):" << endl << x_cur;
            x_prev = x_cur;
        }
        return x_cur;
    }
};



#ifdef WIN32
#define GNUPLOT_NAME "E:\\gnuplot\\bin\\gnuplot -persist"
#else
#define GNUPLOT_NAME "gnuplot -persist"
#endif



void generate_random_points(int m, Matrix& data) {
    default_random_engine _random{std::random_device{}()};
    uniform_real_distribution<double> interval(-50, 50);
    double x, y;
    for (int i = 0; i < m; i++) {
        x = interval(_random);
        y = interval(_random);
        data[i][0] = x;
        data[i][1] = y;
    }
}
int main() {
#ifdef WIN32
    FILE* pipe = _popen(GNUPLOT_NAME, "w");
#else
    FILE* pipe = popen(GNUPLOT_NAME, "w");
#endif
    int m = 30;
    vector<vector<double>> vec_data = {{-49.98, 37.47}, {19.52, 20.22}, {-29.42, -41.01}, {-22.12, 11.02}, {28.75, 18.75},
                                       {43.08, 5.53}, {6.79, -10.31}, {17.51, -11.73}, {-29.78, 16.85}, {16.14, -49.96},
                                       {-35.84, -42.4}, {-28.16, -47.61}, {-21.67, 37.01}, {-26.34, -7.54}, {-20.69, -4.39},
                                       {8.10, -22.52}, {24.23, -10.43}, {-38.68, -40.14}, {-49.76, -9.6}, {13.82, 38.50},
                                       {22.97, -44.44}, {-26.22, -31.32}, {45.77, -48.70}, {24.21, -2.32}, {32.15, -44.87},
                                       {8.33, 32.63}, {-15.59, 26.7}, {-25.99, -38.37}, {47.84, -24.21}, {-29.48, -49.18}};

    int n = 4;
    Matrix data = Matrix(m, 2, vec_data);
//    generate_random_points(m, data);
    Matrix* coefficients = LeastSquareApproximation::doApproximation(m, data, n);
    fprintf(pipe,
            "plot [-50:50][-50:50] %lf*x**4 + %lf*x**3 + %lf*x**2 + %lf*x**1 + %lf*x**0 , '-' using 1:2 with points\n",
            (*coefficients)[4][0], (*coefficients)[3][0], (*coefficients)[2][0], (*coefficients)[1][0], (*coefficients)[0][0]);
    for (int i = 0; i < m; i++) {
        double x, y;
        x = data[i][0];
        y = data[i][1];
        fprintf(pipe, "%f\t%f\n", x, y);
    }
    fprintf(pipe, "e\n");
    fflush(pipe);
#ifdef WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return 0;
}
