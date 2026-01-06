#include <iostream>
#include <iomanip>

#include <variant>

#include <vector>
#include "eigen5/Eigen/Dense"
#include "eigen5/unsupported/Eigen/CXX11/Tensor"
#include <complex>

///*
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <omp.h>

namespace py = pybind11;
//*/

/*
Computing Ewald Sum without splines
*/

using namespace Eigen;
using Complex = std::complex<double>;

template <typename T>
const char* get_type_(const T& x) { return typeid(x).name(); }

template <typename T>
void print_std_(const T& x) { std::cout << x << "\n"; }

template <typename T>
void print_vector_(const std::vector<T>& X, 
                   std::ostream& os = std::cout
                   ) {

    if (X.empty()) {os << "[]\n";}
    else{
    os << "[" ;
    for (size_t i = 0; i < X.size()-1; i++) os << X[i] << ", ";
    os<< X.back();
    os << "]";
    }
}

template <typename T>
void print_shape_(const T& x) {
    std::cout  << "(" << x.rows() << ","<< x.cols() << ")" << "\n";
}

template <typename T>
void print_adress_(const T& x) {
    std::cout << "Adress: " << x.data() << std::endl;
}

template <typename T>
void print_tensor_shape_(const T& x, std::ostream& os = std::cout) {
    auto dims = x.dimensions();
    int rank = dims.rank();
    if (rank == 0) {os << "[]\n";}
    else{
    os << "[" ;
    for (size_t i = 0; i < rank-1; i++) os << dims[i] << ", ";
    os<< x.dimension(rank-1);
    os << "]";
    }
}

using MatrixXd_or_Tensor4d = std::variant<MatrixXd,Tensor<double, 4>>;

MatrixXd_or_Tensor4d make_k_grid_(const std::vector<VectorXd>& marginal_grids, bool flatten = true) {

    // joint_grid_from_marginal_grids_ (list, flatten_output=False)
    int dim = 3; // marginal_grids.size();
    if ((int)marginal_grids.size() != dim) {
        throw std::runtime_error("check input marginal_grids (chould be a three of them)");
    }

    std::vector<int> n_bins;
    std::vector<MatrixXd> Xs = {};
    for (size_t i=0; i<dim; i++) {
        size_t bins_i = marginal_grids[i].rows();
        n_bins.push_back(bins_i);
        MatrixXd x = MatrixXd::Ones(bins_i, dim);
        x.col(i) = marginal_grids[i];
        Xs.push_back(x);
    }

    TensorMap<const Tensor<const double, 2>> tA(Xs[0].data(), n_bins[0], dim);
    TensorMap<const Tensor<const double, 2>> tB(Xs[1].data(), n_bins[1], dim);
    TensorMap<const Tensor<const double, 2>> tC(Xs[2].data(), n_bins[2], dim);

    auto a_4d = tA.shuffle(array<int, 2>{1, 0})
                    .reshape(array<int, 4>{ dim, n_bins[0],         1,         1})
                    .broadcast(array<int, 4>{ 1,         1, n_bins[1], n_bins[2]});

    auto b_4d = tB.shuffle(array<int, 2>{1, 0})
                    .reshape(array<int, 4>{ dim,         1, n_bins[1],         1})
                    .broadcast(array<int, 4>{ 1, n_bins[0],         1, n_bins[2]});

    auto c_4d = tC.shuffle(array<int, 2>{1, 0})
                    .reshape(array<int, 4>{ dim,         1,         1, n_bins[2]})
                    .broadcast(array<int, 4>{ 1, n_bins[0], n_bins[1],         1});

    Tensor<double, 4> result = (a_4d * b_4d * c_4d).eval();
    
    if (not flatten) {
        return result; // (3, n_bins[0], n_bins[1], n_bins[2])
    }
    else {
        if (n_bins[0] != n_bins[1] || n_bins[0] != n_bins[2]) {
            throw std::runtime_error("input marginal_grids should all be the same size");
        }
        Map<const MatrixXd> temp(result.data(), dim, std::pow(n_bins[0],3));
        auto _grid = temp.transpose();
        int i = (int)std::pow(n_bins[0],3)/(int)2;
        int last = _grid.rows() - i - 1;
        MatrixXd grid(_grid.rows()-1, _grid.cols());
        grid << _grid.topRows(i), _grid.bottomRows(last);
        return grid.transpose(); // (3,K) ; K = n_bins[0]**3 - 1
    }
}

/* slow version

class ES_cpp_Eigen_no_einsum { 
public:
    VectorXd q;
    double cutoff;
    int n;
    int N;
    
    double alpha = std::sqrt(-std::log(0.0002)) / cutoff;
    const double PI = 3.141592653589793;
    double pi_over_alpha_sq = std::pow(PI / alpha, 2);

    MatrixXd qq;
    double q_tot;

    double V = 1.0;
    Matrix3d b_inv;
    Matrix3d b_transpose;

    VectorXd k;
    Matrix<double, 3, Dynamic> k_grid;
    Matrix<double, 3, Dynamic> k_grid_rec;
    VectorXd zz;
    VectorXd normalisation;

    Matrix<Complex, Dynamic, Dynamic> pre_abc;
    VectorXd abc;

    MatrixXd s;
    Vector3d ds;
    double d;

    double U_self = 0.0;
    double U_SR   = 0.0;
    double U_LR   = 0.0;
    double U_tot  = 0.0;

    MatrixXd exceptions;
    double exception = 1.0;
    double U_exceptions_correction;

    ES_cpp_Eigen_no_einsum(const VectorXd& charges, double& cut_off, int& bins,
                           const MatrixXd* exceptions_ij = nullptr)
      : q(charges), cutoff(cut_off), n(bins), N(charges.size()) {

        qq = q * q.transpose();
        q_tot = q.sum();
        U_self = - q.dot(q) * alpha / std::sqrt(PI);

        if (exceptions_ij != nullptr) { exceptions = *exceptions_ij;} 
        else { exceptions = MatrixXd::Ones(N,N); }

        k = VectorXd::LinSpaced(2*n+1, -n, n);
        MatrixXd _k_grid = std::get<MatrixXd>(make_k_grid_({k,k,k})); // (3,K)
        k_grid = _k_grid.transpose().topRows(_k_grid.cols()/(int)2).transpose(); // (3,K//2)

        k_grid_rec = k_grid;                 //placeholder
        zz = k_grid.colwise().squaredNorm(); //placeholder
        normalisation = zz;                  //placeholder

        s = MatrixXd::Ones(N,3);             //placeholder
        abc = VectorXd::Zero(k_grid.cols()); //placeholder
    }

    double operator()(const Matrix<double,Dynamic,3>& r, const Matrix3d& b) {

        V = b.determinant();
        b_inv = b.inverse();
        k_grid_rec.noalias() = b_inv*k_grid;
        zz = k_grid_rec.colwise().squaredNorm();
        normalisation.array() = (- pi_over_alpha_sq * zz.array() - zz.array().log()).exp();

        s = r*b_inv;
        pre_abc.array() = (Complex(0,1)*2.0*PI*s*k_grid).array().exp();
        abc.array()     = (q.transpose()*pre_abc).array().abs().square();
        U_LR = normalisation.dot(abc) / (PI*V);

        U_SR = 0.0;
        U_exceptions_correction = 0.0;
        b_transpose = b.transpose();
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                ds = s.row(i) - s.row(j);
                ds -= ds.array().round().matrix();
                d = (b_transpose * ds).norm();
                exception = exceptions(i,j);
                if (d < cutoff && exception > 0.99) {
                    U_SR += (qq(i,j) * std::erfc(alpha * d)) / d;
                }
                if (exception < 0.99) {
                    U_exceptions_correction += qq(i,j) * (exception - std::erf(alpha * d)) / d;
                }
            }
        }
        
        U_tot = U_self + U_SR + U_LR + U_exceptions_correction;
        return U_tot*138.935456; // kJ/mol
    }
};
*/

class ES_cpp_Eigen_einsum {
public:
    VectorXd q;
    double cutoff;
    int n;
    int N;

    double alpha = std::sqrt(-std::log(0.0002)) / cutoff;
    double PI = 3.141592653589793;
    double twoPI = 6.283185307179586;
    double pi_over_alpha_sq = std::pow(PI / alpha, 2);

    MatrixXd qq;
    double q_tot;

    double V;
    Matrix3d b_inv;
    Matrix3d b_transpose;

    VectorXd k;
    int K;
    Tensor<double, 4> k_grid;
    
    array<IndexPair<int>, 1> dims_for_einsum_1 = { IndexPair<int>(1, 0) };
    array<int, 3> dims_for_einsum_2{{2, 1, 0}};

    Tensor<double, 4> k_grid_rec;
    Tensor<double, 3> zz;
    Tensor<double, 3> normalisation;
    Tensor<Complex, 3, ColMajor> pre_abc;
    Tensor<double, 3> abc;
    Tensor<double, 0> abc_normalised_summed;

    MatrixXd s;
    Vector3d ds;
    double d;

    Matrix<Complex, Dynamic, Dynamic> s0k;
    Matrix<Complex, Dynamic, Dynamic> s1k;
    Matrix<Complex, Dynamic, Dynamic> s2k;

    Matrix<Complex, Dynamic, Dynamic, RowMajor> fused;
    Matrix<Complex, Dynamic, Dynamic, RowMajor> result_mat;

    double U_self;
    double U_SR;
    double U_LR;
    double U_tot;

    MatrixXd exceptions;
    double exception = 1.0;
    double U_exceptions_correction;

    ES_cpp_Eigen_einsum(const VectorXd& charges, double& cut_off, int& bins,
                        const MatrixXd* exceptions_ij = nullptr)
        : q(charges), cutoff(cut_off), n(bins), N(charges.size()) {

        qq = q * q.transpose();
        q_tot = q.sum();
        U_self = - q.dot(q) * alpha / std::sqrt(PI);
        
        if (exceptions_ij != nullptr) { exceptions = *exceptions_ij;} 
        else { exceptions = MatrixXd::Ones(N,N); }

        k = VectorXd::LinSpaced(2*n+1, -n, n);
        K = k.rows();

        Tensor<double, 4> output(3, n, n, n);
        k_grid = std::get<Tensor<double, 4>>(make_k_grid_({k,k,k}, false));

        fused = MatrixXd::Zero(N,K*K)*Complex(0,0);
    }

    double operator()(const Matrix<double,Dynamic,3>& r, const Matrix3d& b) {

        V = b.determinant();
        b_inv = b.inverse();

        TensorMap<const Tensor<const double, 2>> b_inv_tensor(b_inv.data(), 3, 3);
        
        //print_std_("k_grid");
        //print_std_(get_type_(k_grid));

        //print_std_("k_grid_rec");
        //print_std_(get_type_(k_grid_rec));
        //print_adress_(k_grid_rec);
        k_grid_rec = b_inv_tensor.contract(k_grid, dims_for_einsum_1);
        //print_std_("k_grid_rec");
        //print_adress_(k_grid_rec);
        //print_std_(get_type_(k_grid_rec));

        //print_std_("zz");
        //print_std_(get_type_(zz));
        //print_adress_(zz);
        zz = k_grid_rec.square().sum(array<int, 1>({0}));
        //print_std_("zz");
        //print_adress_(zz);
        //print_std_(get_type_(zz));

        //print_std_("normalisation");
        //print_std_(get_type_(normalisation));
        //print_adress_(normalisation);
        normalisation = (- pi_over_alpha_sq * zz - zz.log()).exp();
        normalisation(n, n, n) = 0.0;
        //print_std_("normalisation");
        //print_adress_(normalisation);
        //print_std_(get_type_(normalisation));
        //print_tensor_shape_(normalisation);

        s = r*b_inv;

        s0k.array() = (Complex(0,1)*twoPI*s.col(0)*k.transpose()).array().exp(); // (N,K)
        s1k.array() = (Complex(0,1)*twoPI*s.col(1)*k.transpose()).array().exp(); // (N,K)
        s2k.array() = (Complex(0,1)*twoPI*s.col(2)*k.transpose()).array().exp(); // (N,K)

        ///*
        //#pragma omp parallel for schedule(static)
        //#pragma omp atomic
        for (int i = 0; i < N; ++i) {
            Map<Matrix<Complex, Dynamic, Dynamic, RowMajor>>row_mat(fused.data() + i * K * K, K, K);
            row_mat.noalias() = (q(i) * s0k.row(i)).transpose() * s1k.row(i);
        }
        result_mat.noalias() = fused.transpose() * s2k;
        TensorMap<Tensor<Complex, 3, RowMajor>> _pre_abc(result_mat.data(), K, K, K);
        pre_abc = _pre_abc.shuffle(dims_for_einsum_2).swap_layout();
        abc_normalised_summed = (normalisation * pre_abc.abs().square()).sum();
        U_LR = abc_normalised_summed() / (twoPI*V);        
        //abc = pre_abc.abs().square();
        //print_std_("abc");
        //print_std_(abc);
        //*/

        U_SR = 0.0;
        U_exceptions_correction = 0.0;
        ///*
        b_transpose = b.transpose();
        //#pragma omp parallel for reduction(+:U_SR)
        //#pragma omp parallel for schedule(static)
        //#pragma omp for nowait
        //#pragma omp atomic
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                ds = s.row(i) - s.row(j);
                ds -= ds.array().round().matrix();
                d = (b_transpose * ds).norm();
                //d = (b_transpose * ds).squaredNorm();
                //d = std::sqrt(d);
                exception = exceptions(i,j);
                if (d < cutoff && exception > 0.99) {
                    U_SR += (qq(i,j) * std::erfc(alpha * d)) / d;
                }
                if (exception < 0.99) {
                    U_exceptions_correction += qq(i,j) * (exception - std::erf(alpha * d)) / d;
                }
            }
        }
        //*/
        U_tot = U_self + U_SR + U_LR + U_exceptions_correction;
        //print_std_("######## first call done #########");
        return U_tot*138.935456; // kJ/mol
    }
};

// chat gpt code:

using MapMat = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>;

class DOIT {
public:
    VectorXd q;
    double cutoff;
    int n_bins;
    MatrixXd exceptions;
    ES_cpp_Eigen_einsum ES;

    DOIT(Ref<const VectorXd> _q, double _cutoff, int _n_bins,
         Ref<const MatrixXd> exceptions_ij)
        :q(_q), cutoff(_cutoff), n_bins(_n_bins), exceptions(exceptions_ij),
        ES(q, cutoff, n_bins, &exceptions) {}

    void compute_(
                py::array_t<double> r,   // Shape (m, N, 3) positions
                py::array_t<double> b,   // Shape (m, 3, 3) boxes
                Ref<Eigen::VectorXd> out // Shape (m)       output array to write energies
                ) {
        auto bufr = r.request();
        auto bufb = b.request();

        int m = bufr.shape[0];
        int N = bufr.shape[1];
        int d = bufr.shape[2];

        double* ptr_r = static_cast<double*>(bufr.ptr);
        double* ptr_b = static_cast<double*>(bufb.ptr);

        for (int i = 0; i < m; ++i) {
            MapMat pos(ptr_r + (i * N * d), N, d);
            MapMat box(ptr_b + (i * d * d), d, d);
            out(i) = ES(pos, box);
        }
    }
};

PYBIND11_MODULE(eigen_version, m) {
    py::class_<DOIT>(m, "DOIT")
        // could not make exceptions = None as default yet
        // need to pass the (N,N) exceptions_ij array here (all ones means no exceptions)
        .def(py::init<Ref<const VectorXd>, double, int, Ref<const MatrixXd>>())
        .def("compute_", &DOIT::compute_, py::arg("r"), py::arg("b"), py::arg("out"));
}




