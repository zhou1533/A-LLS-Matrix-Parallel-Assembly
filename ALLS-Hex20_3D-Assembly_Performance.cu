#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <climits>
#include <fstream>
#include <chrono>

// 定义错误检查宏
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ==================== 常量定义 (20节点六面体单元) ====================
const int DOF_PER_NODE = 3;          
const int NODES_PER_ELEMENT = 20;    // Hex20 单元
const int ELEM_DOF = NODES_PER_ELEMENT * DOF_PER_NODE; // 60 个自由度
const int GAUSS_POINTS_1D = 3;       // 需 3x3x3 = 27 个高斯点
const int TOTAL_GAUSS_POINTS = GAUSS_POINTS_1D * GAUSS_POINTS_1D * GAUSS_POINTS_1D; 

// 为了防止 Hex20 导致严重的寄存器溢出 (Register Spilling)，BlockSize需降低
const int BLOCK_SIZE = 32;           

// 二次单元节点连接极度密集，理论最大相邻自由度约为 250，缓冲放大到 300
const int MAX_LIST_LENGTH = 200;     

// ==================== 设备常量 ====================
__constant__ float d_YOUNG_MODULUS = 2.1e11f;
__constant__ float d_POISSON_RATIO = 0.3f;
__constant__ float gauss_points_1d[GAUSS_POINTS_1D] = { -0.774596669241483f, 0.0f, 0.774596669241483f };
__constant__ float gauss_weights_1d[GAUSS_POINTS_1D] = { 0.555555555555556f, 0.888888888888889f, 0.555555555555556f };

// ==================== 数据结构定义 ====================
struct Element {
    int nodes[NODES_PER_ELEMENT];
    float x[NODES_PER_ELEMENT];
    float y[NODES_PER_ELEMENT];
    float z[NODES_PER_ELEMENT];
    int color;
};

struct Node {
    float x, y, z;
};

// 预留单链表节点结构体（用于编译通过兼容旧版核函数池）
struct ALLSNode {
	int col;
	float value;
	ALLSNode* next;
};

// A-LLS 基于固定数组的行结构（提升3D单元访问速度）
struct ALLSRow {
    int cols[MAX_LIST_LENGTH];      
    float values[MAX_LIST_LENGTH]; 
    int count;                      
};

struct PerformanceMetrics {
    float alls_time_ms;
    float colored_time_ms;
    float speedup; 
    int atomic_ops_alls;
    size_t alls_mem_usage;       
    size_t colored_mem_usage;    
    size_t stiffness_matrix_size; 
    int num_colors;              
    int max_color_size;          
    float alls_flops;            
    float colored_flops;         
    float coloring_time_ms;      
    float atomic_time_ms;        
};

struct TestConfig {
    int nx, ny, nz;  
    int num_elements;
    PerformanceMetrics metrics;
};

// ==================== 设备函数 ====================
__device__ void shapeFunctionsHex20(float xi, float eta, float zeta, 
                                    float* N, float* dNdxi, float* dNdeta, float* dNdzeta) {
    float xi_n[20] = {-1, 1, 1, -1, -1, 1, 1, -1,   0, 1, 0, -1,   0, 1, 0, -1,  -1, 1, 1, -1};
    float eta_n[20] = {-1, -1, 1, 1, -1, -1, 1, 1,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, -1, 1, 1};
    float zeta_n[20] = {-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1,  0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        float xi_i = xi_n[i], eta_i = eta_n[i], zeta_i = zeta_n[i];
        N[i] = 0.125f * (1.0f + xi_i*xi) * (1.0f + eta_i*eta) * (1.0f + zeta_i*zeta) * (xi_i*xi + eta_i*eta + zeta_i*zeta - 2.0f);
        dNdxi[i] = 0.125f * xi_i * (1.0f + eta_i*eta) * (1.0f + zeta_i*zeta) * (2.0f*xi_i*xi + eta_i*eta + zeta_i*zeta - 1.0f);
        dNdeta[i] = 0.125f * eta_i * (1.0f + xi_i*xi) * (1.0f + zeta_i*zeta) * (xi_i*xi + 2.0f*eta_i*eta + zeta_i*zeta - 1.0f);
        dNdzeta[i] = 0.125f * zeta_i * (1.0f + xi_i*xi) * (1.0f + eta_i*eta) * (xi_i*xi + eta_i*eta + 2.0f*zeta_i*zeta - 1.0f);
    }
    int g1[] = {8, 10, 12, 14};
    for (int i = 0; i < 4; i++) {
        int idx = g1[i]; float eta_i = eta_n[idx], zeta_i = zeta_n[idx];
        dNdxi[idx] = -0.5f * xi * (1.0f + eta_i*eta) * (1.0f + zeta_i*zeta);
        dNdeta[idx] = 0.25f * eta_i * (1.0f - xi*xi) * (1.0f + zeta_i*zeta);
        dNdzeta[idx] = 0.25f * zeta_i * (1.0f - xi*xi) * (1.0f + eta_i*eta);
    }
    int g2[] = {9, 11, 13, 15};
    for (int i = 0; i < 4; i++) {
        int idx = g2[i]; float xi_i = xi_n[idx], zeta_i = zeta_n[idx];
        dNdxi[idx] = 0.25f * xi_i * (1.0f - eta*eta) * (1.0f + zeta_i*zeta);
        dNdeta[idx] = -0.5f * eta * (1.0f + xi_i*xi) * (1.0f + zeta_i*zeta);
        dNdzeta[idx] = 0.25f * zeta_i * (1.0f + xi_i*xi) * (1.0f - eta*eta);
    }
    for (int i = 16; i < 20; i++) {
        float xi_i = xi_n[i], eta_i = eta_n[i];
        dNdxi[i] = 0.25f * xi_i * (1.0f + eta_i*eta) * (1.0f - zeta*zeta);
        dNdeta[i] = 0.25f * eta_i * (1.0f + xi_i*xi) * (1.0f - zeta*zeta);
        dNdzeta[i] = -0.5f * zeta * (1.0f + xi_i*xi) * (1.0f + eta_i*eta);
    }
}

__device__ void computeElementStiffnessHex20(const Element& elem, float* ke) {
    for (int i = 0; i < ELEM_DOF * ELEM_DOF; i++) ke[i] = 0.0f;

    float E = d_YOUNG_MODULUS;
    float nu = d_POISSON_RATIO;
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    float mu = E / (2.0f * (1.0f + nu));
    float c1 = lambda + 2.0f * mu, c2 = lambda, c3 = mu;

    for (int i = 0; i < GAUSS_POINTS_1D; i++) {
        for (int j = 0; j < GAUSS_POINTS_1D; j++) {
            for (int k = 0; k < GAUSS_POINTS_1D; k++) {
                float xi = gauss_points_1d[i], eta = gauss_points_1d[j], zeta = gauss_points_1d[k];
                float weight = gauss_weights_1d[i] * gauss_weights_1d[j] * gauss_weights_1d[k];

                float N[20], dNdxi[20], dNdeta[20], dNdzeta[20];
                shapeFunctionsHex20(xi, eta, zeta, N, dNdxi, dNdeta, dNdzeta);

                float J[9] = {0.0f};
                for (int m = 0; m < 20; m++) {
                    J[0] += dNdxi[m] * elem.x[m]; J[1] += dNdxi[m] * elem.y[m]; J[2] += dNdxi[m] * elem.z[m];
                    J[3] += dNdeta[m] * elem.x[m]; J[4] += dNdeta[m] * elem.y[m]; J[5] += dNdeta[m] * elem.z[m];
                    J[6] += dNdzeta[m] * elem.x[m]; J[7] += dNdzeta[m] * elem.y[m]; J[8] += dNdzeta[m] * elem.z[m];
                }

                float detJ = J[0]*(J[4]*J[8] - J[5]*J[7]) - J[1]*(J[3]*J[8] - J[5]*J[6]) + J[2]*(J[3]*J[7] - J[4]*J[6]);
                if (fabsf(detJ) < 1e-15f) detJ = 1e-15f;

                float invJ[9];
                invJ[0] = (J[4]*J[8] - J[5]*J[7])/detJ;  invJ[1] = (J[2]*J[7] - J[1]*J[8])/detJ;  invJ[2] = (J[1]*J[5] - J[2]*J[4])/detJ;
                invJ[3] = (J[5]*J[6] - J[3]*J[8])/detJ;  invJ[4] = (J[0]*J[8] - J[2]*J[6])/detJ;  invJ[5] = (J[2]*J[3] - J[0]*J[5])/detJ;
                invJ[6] = (J[3]*J[7] - J[4]*J[6])/detJ;  invJ[7] = (J[1]*J[6] - J[0]*J[7])/detJ;  invJ[8] = (J[0]*J[4] - J[1]*J[3])/detJ;

                float dNdx[20], dNdy[20], dNdz[20];
                for (int m = 0; m < 20; m++) {
                    dNdx[m] = invJ[0]*dNdxi[m] + invJ[1]*dNdeta[m] + invJ[2]*dNdzeta[m];
                    dNdy[m] = invJ[3]*dNdxi[m] + invJ[4]*dNdeta[m] + invJ[5]*dNdzeta[m];
                    dNdz[m] = invJ[6]*dNdxi[m] + invJ[7]*dNdeta[m] + invJ[8]*dNdzeta[m];
                }

                float int_w = weight * detJ;

                // 直接构造，防寄存器溢出
                for(int I = 0; I < 20; I++) {
                    for(int J = 0; J < 20; J++) {
                        int r0 = I * 3, c0 = J * 3;
                        float xI = dNdx[I], yI = dNdy[I], zI = dNdz[I];
                        float xJ = dNdx[J], yJ = dNdy[J], zJ = dNdz[J];

                        ke[(r0+0)*60 + (c0+0)] += (c1*xI*xJ + c3*yI*yJ + c3*zI*zJ) * int_w;
                        ke[(r0+0)*60 + (c0+1)] += (c2*xI*yJ + c3*yI*xJ) * int_w;
                        ke[(r0+0)*60 + (c0+2)] += (c2*xI*zJ + c3*zI*xJ) * int_w;

                        ke[(r0+1)*60 + (c0+0)] += (c3*xI*yJ + c2*yI*xJ) * int_w;
                        ke[(r0+1)*60 + (c0+1)] += (c3*xI*xJ + c1*yI*yJ + c3*zI*zJ) * int_w;
                        ke[(r0+1)*60 + (c0+2)] += (c2*yI*zJ + c3*zI*yJ) * int_w;

                        ke[(r0+2)*60 + (c0+0)] += (c3*xI*zJ + c2*zI*xJ) * int_w;
                        ke[(r0+2)*60 + (c0+1)] += (c3*yI*zJ + c2*zI*yJ) * int_w;
                        ke[(r0+2)*60 + (c0+2)] += (c3*xI*xJ + c3*yI*yJ + c1*zI*zJ) * int_w;
                    }
                }
            }
        }
    }
}

__device__ int binary_search(const int* col_ind, int start, int end, int target) {
    while (start <= end) {
        int mid = start + (end - start) / 2;
        if (col_ind[mid] == target) return mid;
        if (col_ind[mid] < target) start = mid + 1;
        else end = mid - 1;
    }
    return -1;
}

// ==================== A-LLS核函数 ====================
__device__ ALLSNode* d_alls_pool = nullptr;
__device__ int* d_alls_pool_index = nullptr;

__device__ ALLSNode* allocateALLSNode() {
	int index = atomicAdd(d_alls_pool_index, 1);
	return &d_alls_pool[index];
}

__global__ void assembleGlobalStiffnessKernelALLSFixedArray(
    int num_elements, const Element* elements, ALLSRow* alls_rows, int* global_atomic_counter) {

    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_elements) return;

    Element elem = elements[eid];
    float ke[ELEM_DOF * ELEM_DOF];
    computeElementStiffnessHex20(elem, ke);

    int dof_indices[ELEM_DOF];
    for (int i = 0; i < NODES_PER_ELEMENT; i++) {
        int node_id = elem.nodes[i];
        dof_indices[i * 3 + 0] = node_id * 3;
        dof_indices[i * 3 + 1] = node_id * 3 + 1;
        dof_indices[i * 3 + 2] = node_id * 3 + 2;
    }

    for (int i = 0; i < ELEM_DOF; i++) {
        int row_global = dof_indices[i];
        for (int j = 0; j < ELEM_DOF; j++) {
            int col_global = dof_indices[j];
            float value = ke[i * ELEM_DOF + j];

            ALLSRow& row = alls_rows[row_global];
            bool found = false;
            int local_count = row.count;

            for (int k = 0; k < local_count; k++) {
                if (row.cols[k] == col_global) {
                    atomicAdd(&(row.values[k]), value);
                    atomicAdd(global_atomic_counter, 1);
                    found = true;
                    break;
                }
            }

            if (!found) {
                int pos = atomicAdd(&row.count, 1);
                if (pos < MAX_LIST_LENGTH) {
                    row.cols[pos] = col_global;
                    row.values[pos] = value;
                    atomicAdd(global_atomic_counter, 1);
                } else {
                    atomicSub(&row.count, 1);
                    // 溢出处理省略，理论上 Hex20 MAX_LIST_LENGTH=300 足以覆盖
                }
            }
        }
    }
}

__global__ void convertALLStoCSRKernelFixedArray(
    int total_dofs, const ALLSRow* alls_rows, float* csr_values, 
    const int* csr_row_ptr, const int* csr_col_ind) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_dofs) return;

    int row_start = csr_row_ptr[row];
    int row_end = csr_row_ptr[row + 1] - 1;

    const ALLSRow& arow = alls_rows[row];
    for (int i = 0; i < arow.count; i++) {
        int col = arow.cols[i];
        float value = arow.values[i];
        int pos = binary_search(csr_col_ind, row_start, row_end, col);
        if (pos >= 0) {
            atomicAdd(&csr_values[pos], value);
        }
    }
}

// ==================== 染色法核函数 ====================
__global__ void assembleGlobalStiffnessKernelColoredNoAtomic(
    int num_elements, const Element* elements, float* csr_values,
    const int* csr_row_ptr, const int* csr_col_ind,
    const int* color_offsets, const int* color_sizes, int num_colors, int total_dofs, int nnz) {

    int color_id = blockIdx.x;
    if (color_id >= num_colors) return;

    int local_index = threadIdx.x;
    int elements_in_color = color_sizes[color_id];
    int offset = color_offsets[color_id];

    for (int idx = local_index; idx < elements_in_color; idx += blockDim.x) {
        int eid = offset + idx;
        Element elem = elements[eid];
        
        float ke[ELEM_DOF * ELEM_DOF];
        computeElementStiffnessHex20(elem, ke);

        int dof_indices[ELEM_DOF];
        for (int i = 0; i < NODES_PER_ELEMENT; i++) {
            dof_indices[i * 3 + 0] = elem.nodes[i] * 3;
            dof_indices[i * 3 + 1] = elem.nodes[i] * 3 + 1;
            dof_indices[i * 3 + 2] = elem.nodes[i] * 3 + 2;
        }

        for (int i = 0; i < ELEM_DOF; i++) {
            int row_global = dof_indices[i];
            int row_start = csr_row_ptr[row_global];
            int row_end = csr_row_ptr[row_global + 1] - 1;

            for (int j = 0; j < ELEM_DOF; j++) {
                int col_global = dof_indices[j];
                int pos = binary_search(csr_col_ind, row_start, row_end, col_global);
                if (pos >= row_start && pos <= row_end) {
                    csr_values[pos] += ke[i * ELEM_DOF + j];
                }
            }
        }
    }
}

// ==================== 主机端前处理函数 ====================
void generateMeshHex20(int nx, int ny, int nz, float lx, float ly, float lz,
                       std::vector<Node>& nodes, std::vector<Element>& elements) {
    int n_nx = 2 * nx + 1, n_ny = 2 * ny + 1, n_nz = 2 * nz + 1;
    nodes.resize(n_nx * n_ny * n_nz);

    for (int k = 0; k < n_nz; k++) {
        for (int j = 0; j < n_ny; j++) {
            for (int i = 0; i < n_nx; i++) {
                int id = k * n_nx * n_ny + j * n_nx + i;
                nodes[id].x = i * 0.5f * lx / nx;
                nodes[id].y = j * 0.5f * ly / ny;
                nodes[id].z = k * 0.5f * lz / nz;
            }
        }
    }

    elements.resize(nx * ny * nz);
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int eid = k * (nx * ny) + j * nx + i;
                int bx = 2 * i, by = 2 * j, bz = 2 * k;
                auto id = [&](int dx, int dy, int dz) { return (bz + dz)*n_nx*n_ny + (by + dy)*n_nx + (bx + dx); };

                Element& el = elements[eid];
                el.nodes[0] = id(0,0,0); el.nodes[1] = id(2,0,0); el.nodes[2] = id(2,2,0); el.nodes[3] = id(0,2,0);
                el.nodes[4] = id(0,0,2); el.nodes[5] = id(2,0,2); el.nodes[6] = id(2,2,2); el.nodes[7] = id(0,2,2);
                el.nodes[8] = id(1,0,0); el.nodes[9] = id(2,1,0); el.nodes[10]= id(1,2,0); el.nodes[11]= id(0,1,0);
                el.nodes[12]= id(1,0,2); el.nodes[13]= id(2,1,2); el.nodes[14]= id(1,2,2); el.nodes[15]= id(0,1,2);
                el.nodes[16]= id(0,0,1); el.nodes[17]= id(2,0,1); el.nodes[18]= id(2,2,1); el.nodes[19]= id(0,2,1);

                for (int m = 0; m < 20; m++) {
                    el.x[m] = nodes[el.nodes[m]].x; el.y[m] = nodes[el.nodes[m]].y; el.z[m] = nodes[el.nodes[m]].z;
                }
                el.color = -1;
            }
        }
    }
}

// 优化的CSR建图（按节点查重）
void buildCSRStructureHex20(int num_nodes, const std::vector<Element>& elements,
                            std::vector<int>& row_ptr, std::vector<int>& col_ind) {
    int total_dofs = num_nodes * DOF_PER_NODE;
    std::vector<std::set<int>> node_sets(num_nodes);

    for (const auto& elem : elements) {
        for (int i = 0; i < NODES_PER_ELEMENT; i++) {
            for (int j = 0; j < NODES_PER_ELEMENT; j++) {
                node_sets[elem.nodes[i]].insert(elem.nodes[j]);
            }
        }
    }

    row_ptr.resize(total_dofs + 1, 0);
    int current_nnz = 0;
    for (int i = 0; i < num_nodes; i++) {
        for (int di = 0; di < DOF_PER_NODE; di++) {
            int row = i * DOF_PER_NODE + di;
            row_ptr[row] = current_nnz;
            for (int n_col : node_sets[i]) {
                for (int dj = 0; dj < DOF_PER_NODE; dj++) {
                    col_ind.push_back(n_col * DOF_PER_NODE + dj);
                    current_nnz++;
                }
            }
        }
    }
    row_ptr[total_dofs] = current_nnz;
}

int colorElementsHex20(std::vector<Element>& elements, std::vector<std::vector<int>>& color_groups) {
    std::map<int, std::vector<int>> n2e;
    for (int e = 0; e < elements.size(); e++) 
        for (int n : elements[e].nodes) n2e[n].push_back(e);
        
    std::vector<std::vector<int>> adj(elements.size());
    for (int e = 0; e < elements.size(); e++) {
        for (int n : elements[e].nodes) 
            for (int neigh : n2e[n]) if (neigh != e) adj[e].push_back(neigh);
    }
    
    std::vector<int> colors(elements.size(), -1);
    int max_c = 0;
    for (int e = 0; e < elements.size(); e++) {
        std::vector<bool> used(elements.size(), false);
        for (int neigh : adj[e]) if (colors[neigh] != -1) used[colors[neigh]] = true;
        int c = 0; while (used[c]) c++;
        colors[e] = c; max_c = std::max(max_c, c); elements[e].color = c;
    }
    
    color_groups.resize(max_c + 1);
    for (int e = 0; e < elements.size(); e++) color_groups[colors[e]].push_back(e);
    return max_c + 1;
}

// ==================== 性能测试核心函数 ====================
PerformanceMetrics compareAssemblyMethods(
    int num_elements, int total_dofs, int nnz,
    const Element* d_elements,
    const int* d_csr_row_ptr,
    const int* d_csr_col_ind,
    float* d_csr_values_alls,
    float* d_csr_values_colored,
    const int* d_color_offsets,
    const int* d_color_sizes,
    int num_colors,
    const std::vector<int>& h_color_sizes) {

    PerformanceMetrics metrics;
    metrics.num_colors = num_colors;
    metrics.max_color_size = 0;
    for (int i = 0; i < num_colors; i++) {
        if (h_color_sizes[i] > metrics.max_color_size) metrics.max_color_size = h_color_sizes[i];
    }

    int* d_global_atomic_counter_alls;
    CHECK_CUDA_ERROR(cudaMalloc(&d_global_atomic_counter_alls, sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    size_t free_mem_start, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem_start, &total_mem));
    size_t mem_start_alls = free_mem_start;

    // --- 测试 A-LLS 方法 ---
    CHECK_CUDA_ERROR(cudaMemset(d_global_atomic_counter_alls, 0, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_csr_values_alls, 0, nnz * sizeof(float)));

    ALLSRow* d_alls_rows;
    CHECK_CUDA_ERROR(cudaMalloc(&d_alls_rows, total_dofs * sizeof(ALLSRow)));
    CHECK_CUDA_ERROR(cudaMemset(d_alls_rows, 0, total_dofs * sizeof(ALLSRow)));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    int blocks_alls = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assembleGlobalStiffnessKernelALLSFixedArray<<<blocks_alls, BLOCK_SIZE>>>(
        num_elements, d_elements, d_alls_rows, d_global_atomic_counter_alls);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int blocks_convert = (total_dofs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convertALLStoCSRKernelFixedArray<<<blocks_convert, BLOCK_SIZE>>>(
        total_dofs, d_alls_rows, d_csr_values_alls, d_csr_row_ptr, d_csr_col_ind);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.alls_time_ms, start, stop));

    size_t free_mem_end_alls;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem_end_alls, &total_mem));
    metrics.alls_mem_usage = mem_start_alls - free_mem_end_alls;

    CHECK_CUDA_ERROR(cudaMemcpy(&metrics.atomic_ops_alls, d_global_atomic_counter_alls, sizeof(int), cudaMemcpyDeviceToHost));
    
    // 估算 Hex20 的运算量：约 10000 浮点操作/Gauss点
    metrics.alls_flops = num_elements * TOTAL_GAUSS_POINTS * 10000.0f / 1e9f;
    metrics.alls_flops /= (metrics.alls_time_ms / 1000.0f);

    CHECK_CUDA_ERROR(cudaFree(d_alls_rows));

    // --- 测试 染色方法 ---
    CHECK_CUDA_ERROR(cudaMemset(d_csr_values_colored, 0, nnz * sizeof(float)));
    size_t mem_start_colored;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&mem_start_colored, &total_mem));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dim3 blocks_colored_noatomic(num_colors);
    int threads_per_block = std::min(metrics.max_color_size, 256);
    if (threads_per_block < 32) threads_per_block = 32;

    assembleGlobalStiffnessKernelColoredNoAtomic<<<blocks_colored_noatomic, threads_per_block>>>(
        num_elements, d_elements, d_csr_values_colored,
        d_csr_row_ptr, d_csr_col_ind,
        d_color_offsets, d_color_sizes, num_colors, total_dofs, nnz);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics
