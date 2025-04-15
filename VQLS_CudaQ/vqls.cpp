#include "cudaq.h"

#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/vqe.h"
#include <iostream>

struct hardware_efficient_ansatz{
    void operator()(std::<vector> vAngles, int nQubits) __qpu__{
        cudaq::qvector ans(nQubits);
        int nLayers = vAngles.size();
        int idx = 0;

        nLayers = nLayers/(2 * nQubits);
        for (int i = 0; i < nLayers; i++)
        {
            for (int j = 0; j < nQubits; j++)
            {
                idx = i * nQubits + (2 * j);
                ry(vAngles[idx], ans(j));
                rx(vAngles[idx + 1], ans(j));
                if (j < (nQubits - 1))
                {
                    x::<cudaq::ctrl>(ans(j+1), ans(j));
                }

            }
        }

    }
}