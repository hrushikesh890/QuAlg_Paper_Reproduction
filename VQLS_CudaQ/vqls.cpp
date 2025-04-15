#include "cudaq.h"
#include "cudaq/algorithms/observe.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/gradient.h"

#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>

const int nQubits = 4;
const int nLayers = 8;

std::vector<std::pair<std::complex<double>, std::string>> matA = {
    {{1.0, 0.0}, "IZZI"},
    {{2.0, 0.0}, "ZZZZ"},
    {{-0.5, 0.0}, "IIIZ"}
};

// Hardware-efficient ansatz
struct hardware_efficient_ansatz {
    void operator()(std::vector<double> theta) __qpu__ {
        cudaq::qvector<> q(nQubits);
        int idx = 0;
        for (int i = 0; i < nLayers; ++i) {
            for (int j = 0; j < nQubits - 1; j += 2) {
                ry(theta[idx++], q[j]);
                ry(theta[idx++], q[j + 1]);
                x<cudaq::ctrl>(q[j + 1], q[j]);
            }
        }
    }
};

// Prepare |b⟩ = H^⊗n |0⟩
struct state_b_kernel {
    void operator()() __qpu__ {
        cudaq::qvector<> q(nQubits);
        for (int i = 0; i < q.size(); ++i)
            h(q[i]);
    }
};

// Convert string-based Pauli terms to spin_op
cudaq::spin_op to_spin_op(const std::vector<std::pair<std::complex<double>, std::string>>& pauli_list) {
    cudaq::spin_op H;
    for (const auto& [coeff, pauli_str] : pauli_list) {
        cudaq::spin_op term(1.0);
        for (std::size_t i = 0; i < pauli_str.size(); ++i) {
            switch (pauli_str[i]) {
                case 'I': break;
                case 'X': term *= cudaq::spin::x(i); break;
                case 'Y': term *= cudaq::spin::y(i); break;
                case 'Z': term *= cudaq::spin::z(i); break;
                default:
                    throw std::invalid_argument("Invalid Pauli character: " + std::string(1, pauli_str[i]));
            }
        }
        H += coeff * term;
    }
    return H;
}

// VQLS cost function
double vqls_cost(std::vector<double> theta) {
    auto A = to_spin_op(matA);
    auto AdgA = A * A;

    // 1. ⟨ψ|A†A|ψ⟩
    double denom = cudaq::observe(hardware_efficient_ansatz{}, AdgA, theta).expectation();

    // 2. Approximate ⟨b|A|ψ⟩ using ⟨ψ|A|b⟩ by running |b⟩ through A and observing
    double numer = cudaq::observe(
        []() __qpu__ {
            cudaq::qvector<> q(nQubits);
            for (int i = 0; i < q.size(); ++i) h(q[i]);
        },
        A
    ).expectation();

    return denom - 2.0 * numer + 1.0;
}

int main() {
    std::vector<double> theta(nQubits * nLayers, 0.0);

    auto cost_fn = [](std::vector<double> th) { return vqls_cost(th); };
    auto grad_fn = cudaq::gradient(cost_fn);

    for (int iter = 0; iter < 50; ++iter) {
        auto grad = grad_fn(theta);
        for (int i = 0; i < theta.size(); ++i)
            theta[i] -= 0.1 * grad[i];

        std::cout << "Iter " << iter << " Cost = " << vqls_cost(theta) << "\n";
    }

    std::cout << "Final parameters:\n";
    for (auto val : theta) std::cout << val << " ";
    std::cout << "\n";

    return 0;
}
