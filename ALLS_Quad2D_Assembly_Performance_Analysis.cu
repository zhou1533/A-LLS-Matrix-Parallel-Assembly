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
#include <list>
#include <forward_list>
#include <fstream>
#include <ctime>
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

// ==================== 常量定义 ====================
const int DOF_PER_NODE = 2;
const int NODES_PER_ELEMENT = 4;
const int ELEM_DOF = NODES_PER_ELEMENT * DOF_PER_NODE;
const int GAUSS_POINTS = 2;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;
const int MAX_LIST_LENGTH = 32; // 链表最大长度

// ==================== 测试控制常量 ====================
const int WARMUP_ITERS = 3;   // 预热次数，消除显卡唤醒/缓存冷启动时间
const int MEASURE_ITERS = 10; // 测量次数，求平均值以平滑性能波动

// ==================== 设备常量 ====================
__constant__ double d_YOUNG_MODULUS = 2.1e11;
__constant__ double d_POISSON_RATIO = 0.3;
__constant__ double d_THICKNESS = 0.01;

// ==================== 主机端常量 ====================
const double h_YOUNG_MODULUS = 2.1e11;
const double h_POISSON_RATIO = 0.3;
const double h_THICKNESS = 0.01;
const double h_gauss_points[GAUSS_POINTS] = { -0.577350269189626, 0.577350269189626 };
const double h_gauss_weights[GAUSS_POINTS] = { 1.0, 1.0 };

// ==================== 数据结构定义 ====================
struct Element {
	int nodes[NODES_PER_ELEMENT];
	double x[NODES_PER_ELEMENT];
	double y[NODES_PER_ELEMENT];
	int color;  // 用于染色算法
};

struct Node {
	double x, y;
};

// A-LLS数据结构
struct ALLSNode {
	int col;
	double value;
	ALLSNode* next;
};

// 性能分析结构体
struct PerformanceMetrics {
	float alls_time_ms;
	float colored_time_ms;
	float speedup; // 染色方法相对于A-LLS的加速比
	int atomic_ops_alls;

	// 新增指标
	size_t alls_mem_usage;       // A-LLS方法GPU内存使用量（字节）
	size_t colored_mem_usage;    // 染色方法GPU内存使用量（字节）
	size_t stiffness_matrix_size; // 刚度矩阵内存大小（字节）
	int num_colors;              // 染色数量
	int max_color_size;          // 最大颜色组大小
	float alls_flops;            // A-LLS方法浮点运算性能（GFLOPS）
	float colored_flops;         // 染色方法浮点运算性能（GFLOPS）
	float coloring_time_ms;      // 主机端染色时间（毫秒）
	float atomic_time_ms;        // 原子操作估算时间（毫秒）
};

// 测试配置结构体
struct TestConfig {
	int nx;
	int ny;
	int num_elements;
	PerformanceMetrics metrics;
};

// ==================== GPU常量内存 ====================
__constant__ double gauss_points[GAUSS_POINTS] = { -0.577350269189626, 0.577350269189626 };
__constant__ double gauss_weights[GAUSS_POINTS] = { 1.0, 1.0 };

// ==================== 原子操作支持 ====================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

// ==================== 设备函数 ====================
__device__ void shapeFunctions(double xi, double eta, double* N, double* dNdxi, double* dNdeta) {
	N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
	N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
	N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
	N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

	dNdxi[0] = -0.25 * (1.0 - eta);
	dNdxi[1] = 0.25 * (1.0 - eta);
	dNdxi[2] = 0.25 * (1.0 + eta);
	dNdxi[3] = -0.25 * (1.0 + eta);

	dNdeta[0] = -0.25 * (1.0 - xi);
	dNdeta[1] = -0.25 * (1.0 + xi);
	dNdeta[2] = 0.25 * (1.0 + xi);
	dNdeta[3] = 0.25 * (1.0 - xi);
}

__device__ void computeJacobianAndBMatrix(const double* dNdxi, const double* dNdeta,
	const double* x, const double* y,
	double* J, double& detJ, double* B) {

	J[0] = 0.0; J[1] = 0.0;
	J[2] = 0.0; J[3] = 0.0;

	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		J[0] += dNdxi[i] * x[i];
		J[1] += dNdxi[i] * y[i];
		J[2] += dNdeta[i] * x[i];
		J[3] += dNdeta[i] * y[i];
	}

	detJ = J[0] * J[3] - J[1] * J[2];
	if (fabs(detJ) < 1e-10) detJ = 1e-10;

	double invJ[4];
	invJ[0] = J[3] / detJ;
	invJ[1] = -J[1] / detJ;
	invJ[2] = -J[2] / detJ;
	invJ[3] = J[0] / detJ;

	double dNdx[4], dNdy[4];
	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		dNdx[i] = invJ[0] * dNdxi[i] + invJ[1] * dNdeta[i];
		dNdy[i] = invJ[2] * dNdxi[i] + invJ[3] * dNdeta[i];
	}

	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		int col = i * 2;
		B[0 * ELEM_DOF + col] = dNdx[i];
		B[0 * ELEM_DOF + col + 1] = 0.0;
		B[1 * ELEM_DOF + col] = 0.0;
		B[1 * ELEM_DOF + col + 1] = dNdy[i];
		B[2 * ELEM_DOF + col] = dNdy[i];
		B[2 * ELEM_DOF + col + 1] = dNdx[i];
	}
}

__device__ void computeDMatrix(double* D) {
	double E = d_YOUNG_MODULUS;
	double nu = d_POISSON_RATIO;
	double factor = E / (1.0 - nu * nu);

	D[0] = factor;         D[1] = factor * nu;   D[2] = 0.0;
	D[3] = factor * nu;    D[4] = factor;        D[5] = 0.0;
	D[6] = 0.0;           D[7] = 0.0;          D[8] = factor * (1.0 - nu) / 2.0;
}

__device__ void computeElementStiffness(const Element& elem, double* ke) {
	for (int i = 0; i < ELEM_DOF * ELEM_DOF; i++) ke[i] = 0.0;

	double D[9];
	computeDMatrix(D);

	for (int i = 0; i < GAUSS_POINTS; i++) {
		for (int j = 0; j < GAUSS_POINTS; j++) {
			double xi = gauss_points[i];
			double eta = gauss_points[j];
			double weight = gauss_weights[i] * gauss_weights[j];

			double N[4], dNdxi[4], dNdeta[4];
			shapeFunctions(xi, eta, N, dNdxi, dNdeta);

			double J[4], detJ;
			double B[3 * ELEM_DOF] = { 0.0 };
			computeJacobianAndBMatrix(dNdxi, dNdeta, elem.x, elem.y, J, detJ, B);

			double integration_weight = weight * detJ * d_THICKNESS;

			for (int m = 0; m < ELEM_DOF; m++) {
				for (int n = 0; n < ELEM_DOF; n++) {
					double sum = 0.0;
					for (int k = 0; k < 3; k++) {
						for (int l = 0; l < 3; l++) {
							sum += B[k * ELEM_DOF + m] * D[k * 3 + l] * B[l * ELEM_DOF + n];
						}
					}
					ke[m * ELEM_DOF + n] += sum * integration_weight;
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

// ==================== 主机端函数 ====================
void shapeFunctionsHost(double xi, double eta, double* N, double* dNdxi, double* dNdeta) {
	N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
	N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
	N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
	N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

	dNdxi[0] = -0.25 * (1.0 - eta);
	dNdxi[1] = 0.25 * (1.0 - eta);
	dNdxi[2] = 0.25 * (1.0 + eta);
	dNdxi[3] = -0.25 * (1.0 + eta);

	dNdeta[0] = -0.25 * (1.0 - xi);
	dNdeta[1] = -0.25 * (1.0 + xi);
	dNdeta[2] = 0.25 * (1.0 + xi);
	dNdeta[3] = 0.25 * (1.0 - xi);
}

void computeJacobianAndBMatrixHost(const double* dNdxi, const double* dNdeta,
	const double* x, const double* y,
	double* J, double& detJ, double* B) {

	J[0] = 0.0; J[1] = 0.0;
	J[2] = 0.0; J[3] = 0.0;

	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		J[0] += dNdxi[i] * x[i];
		J[1] += dNdxi[i] * y[i];
		J[2] += dNdeta[i] * x[i];
		J[3] += dNdeta[i] * y[i];
	}

	detJ = J[0] * J[3] - J[1] * J[2];
	if (fabs(detJ) < 1e-10) detJ = 1e-10;

	double invJ[4];
	invJ[0] = J[3] / detJ;
	invJ[1] = -J[1] / detJ;
	invJ[2] = -J[2] / detJ;
	invJ[3] = J[0] / detJ;

	double dNdx[4], dNdy[4];
	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		dNdx[i] = invJ[0] * dNdxi[i] + invJ[1] * dNdeta[i];
		dNdy[i] = invJ[2] * dNdxi[i] + invJ[3] * dNdeta[i];
	}

	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		int col = i * 2;
		B[0 * ELEM_DOF + col] = dNdx[i];
		B[0 * ELEM_DOF + col + 1] = 0.0;
		B[1 * ELEM_DOF + col] = 0.0;
		B[1 * ELEM_DOF + col + 1] = dNdy[i];
		B[2 * ELEM_DOF + col] = dNdy[i];
		B[2 * ELEM_DOF + col + 1] = dNdx[i];
	}
}

void computeDMatrixHost(double* D) {
	double E = h_YOUNG_MODULUS;
	double nu = h_POISSON_RATIO;
	double factor = E / (1.0 - nu * nu);

	D[0] = factor;         D[1] = factor * nu;   D[2] = 0.0;
	D[3] = factor * nu;    D[4] = factor;        D[5] = 0.0;
	D[6] = 0.0;           D[7] = 0.0;          D[8] = factor * (1.0 - nu) / 2.0;
}

void computeElementStiffnessHost(const Element& elem, double* ke) {
	for (int i = 0; i < ELEM_DOF * ELEM_DOF; i++) ke[i] = 0.0;

	double D[9];
	computeDMatrixHost(D);

	for (int i = 0; i < GAUSS_POINTS; i++) {
		for (int j = 0; j < GAUSS_POINTS; j++) {
			double xi = h_gauss_points[i];
			double eta = h_gauss_points[j];
			double weight = h_gauss_weights[i] * h_gauss_weights[j];

			double N[4], dNdxi[4], dNdeta[4];
			shapeFunctionsHost(xi, eta, N, dNdxi, dNdeta);

			double J[4], detJ;
			double B[3 * ELEM_DOF] = { 0.0 };
			computeJacobianAndBMatrixHost(dNdxi, dNdeta, elem.x, elem.y, J, detJ, B);

			double integration_weight = weight * detJ * h_THICKNESS;

			for (int m = 0; m < ELEM_DOF; m++) {
				for (int n = 0; n < ELEM_DOF; n++) {
					double sum = 0.0;
					for (int k = 0; k < 3; k++) {
						for (int l = 0; l < 3; l++) {
							sum += B[k * ELEM_DOF + m] * D[k * 3 + l] * B[l * ELEM_DOF + n];
						}
					}
					ke[m * ELEM_DOF + n] += sum * integration_weight;
				}
			}
		}
	}
}

// 主机端组装全局刚度矩阵
void assembleGlobalStiffnessHost(int num_elements, const std::vector<Element>& elements,
	const std::vector<int>& csr_row_ptr, const std::vector<int>& csr_col_ind,
	std::vector<double>& csr_values) {

	// 清零CSR值数组
	std::fill(csr_values.begin(), csr_values.end(), 0.0);

	for (const auto& elem : elements) {
		double ke[ELEM_DOF * ELEM_DOF];
		computeElementStiffnessHost(elem, ke);

		int dof_indices[ELEM_DOF];
		for (int i = 0; i < NODES_PER_ELEMENT; i++) {
			int node_id = elem.nodes[i];
			dof_indices[i * 2] = node_id * 2;
			dof_indices[i * 2 + 1] = node_id * 2 + 1;
		}

		for (int i = 0; i < ELEM_DOF; i++) {
			int row_global = dof_indices[i];
			int row_start = csr_row_ptr[row_global];
			int row_end = csr_row_ptr[row_global + 1] - 1;

			for (int j = 0; j < ELEM_DOF; j++) {
				int col_global = dof_indices[j];

				// 在CSR列索引中使用二分查找找到对应的存储位置
				int low = row_start;
				int high = row_end;
				int pos = -1;

				while (low <= high) {
					int mid = low + (high - low) / 2;
					if (csr_col_ind[mid] == col_global) {
						pos = mid;
						break;
					}
					else if (csr_col_ind[mid] < col_global) {
						low = mid + 1;
					}
					else {
						high = mid - 1;
					}
				}

				if (pos >= 0) {
					csr_values[pos] += ke[i * ELEM_DOF + j];
				}
			}
		}
	}
}

// ==================== A-LLS数据结构管理 ====================
// 预分配的A-LLS节点池
__device__ ALLSNode* d_alls_pool = nullptr;
__device__ int* d_alls_pool_index = nullptr;

// 获取一个新的A-LLS节点
__device__ ALLSNode* allocateALLSNode() {
	int index = atomicAdd(d_alls_pool_index, 1);
	return &d_alls_pool[index];
}

// ==================== 基于A-LLS的优化核函数 ====================
__global__ void assembleGlobalStiffnessKernelALLS(
	int num_elements,
	const Element* elements,
	ALLSNode** alls_row_heads,
	int* global_atomic_counter) {

	int eid = blockIdx.x * blockDim.x + threadIdx.x;
	if (eid >= num_elements) return;

	Element elem = elements[eid];
	double ke[ELEM_DOF * ELEM_DOF];
	computeElementStiffness(elem, ke);

	int dof_indices[ELEM_DOF];
	for (int i = 0; i < NODES_PER_ELEMENT; i++) {
		int node_id = elem.nodes[i];
		dof_indices[i * 2] = node_id * 2;
		dof_indices[i * 2 + 1] = node_id * 2 + 1;
	}

	for (int i = 0; i < ELEM_DOF; i++) {
		int row_global = dof_indices[i];
		if (row_global < 0) continue;

		for (int j = 0; j < ELEM_DOF; j++) {
			int col_global = dof_indices[j];
			if (col_global < 0) continue;

			double value = ke[i * ELEM_DOF + j];

			// 获取行头指针
			ALLSNode* head = alls_row_heads[row_global];

			// 查找是否已有该列
			ALLSNode* current = head;
			ALLSNode* prev = nullptr;
			bool found = false;

			while (current != nullptr) {
				if (current->col == col_global) {
					// 找到相同列，累加值
					atomicAdd(&(current->value), value);
					atomicAdd(global_atomic_counter, 1);
					found = true;
					break;
				}
				prev = current;
				current = current->next;
			}

			if (!found) {
				// 创建新节点
				ALLSNode* newNode = allocateALLSNode();
				newNode->col = col_global;
				newNode->value = value;
				newNode->next = nullptr;

				// 插入到链表末尾
				if (prev == nullptr) {
					// 空链表，使用原子操作设置头指针
					ALLSNode* oldHead = (ALLSNode*)atomicExch((unsigned long long int*)&alls_row_heads[row_global],
						(unsigned long long int)newNode);
					newNode->next = oldHead;
				}
				else {
					// 非空链表，使用原子操作插入到prev后面
					ALLSNode* oldNext = (ALLSNode*)atomicExch((unsigned long long int*)&prev->next,
						(unsigned long long int)newNode);
					newNode->next = oldNext;
				}
				atomicAdd(global_atomic_counter, 1);
			}
		}
	}
}

// ==================== 将A-LLS转换为CSR ====================
__global__ void convertALLStoCSRKernel(
	int total_dofs,
	ALLSNode** alls_row_heads,
	double* csr_values,
	const int* csr_row_ptr,
	const int* csr_col_ind) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= total_dofs) return;

	int row_start = csr_row_ptr[row];
	int row_end = csr_row_ptr[row + 1] - 1;

	// 注：现在由于存在多轮测试，行清零工作移到主机端的cudaMemset进行

	// 累加A-LLS中的值
	ALLSNode* current = alls_row_heads[row];
	while (current != nullptr) {
		int col = current->col;
		double value = current->value;

		// 在CSR中找到对应位置
		int pos = binary_search(csr_col_ind, row_start, row_end, col);
		if (pos >= 0) {
			atomicAdd(&csr_values[pos], value); // 使用原子加确保正确性
		}

		current = current->next;
	}
}

// ==================== 基于染色的优化核函数（无原子操作） ====================
__global__ void assembleGlobalStiffnessKernelColoredNoAtomic(
	int num_elements,
	const Element* elements,
	double* csr_values,
	const int* csr_row_ptr,
	const int* csr_col_ind,
	const int* color_offsets,
	const int* color_sizes,
	int num_colors,
	int total_dofs,
	int nnz) {

	int color_id = blockIdx.x;
	if (color_id >= num_colors) return;

	int local_index = threadIdx.x;
	int elements_in_color = color_sizes[color_id];

	if (color_offsets[color_id] < 0 || color_offsets[color_id] >= num_elements) {
		return;
	}

	if (color_offsets[color_id] + elements_in_color > num_elements) {
		elements_in_color = num_elements - color_offsets[color_id];
	}

	for (int idx = local_index; idx < elements_in_color; idx += blockDim.x) {
		int eid = color_offsets[color_id] + idx;
		if (eid < 0 || eid >= num_elements) continue;

		Element elem = elements[eid];
		double ke[ELEM_DOF * ELEM_DOF];
		computeElementStiffness(elem, ke);

		int dof_indices[ELEM_DOF];
		for (int i = 0; i < NODES_PER_ELEMENT; i++) {
			int node_id = elem.nodes[i];
			if (node_id < 0) {
				dof_indices[i * 2] = -1;
				dof_indices[i * 2 + 1] = -1;
				continue;
			}
			dof_indices[i * 2] = node_id * 2;
			dof_indices[i * 2 + 1] = node_id * 2 + 1;
		}

		for (int i = 0; i < ELEM_DOF; i++) {
			int row_global = dof_indices[i];
			if (row_global < 0 || row_global >= total_dofs) continue;

			int row_start = csr_row_ptr[row_global];
			int row_end = csr_row_ptr[row_global + 1] - 1;

			if (row_start < 0 || row_end >= nnz || row_start > row_end) continue;

			for (int j = 0; j < ELEM_DOF; j++) {
				int col_global = dof_indices[j];
				if (col_global < 0 || col_global >= total_dofs) continue;

				int pos = binary_search(csr_col_ind, row_start, row_end, col_global);
				if (pos >= row_start && pos <= row_end) {
					csr_values[pos] += ke[i * ELEM_DOF + j];
				}
			}
		}
	}

	__syncthreads();
}

// ==================== 主机端辅助函数 ====================
void generateMesh(int nx, int ny, double lx, double ly,
	std::vector<Node>& nodes, std::vector<Element>& elements) {
	nodes.resize((nx + 1) * (ny + 1));
	for (int j = 0; j <= ny; j++) {
		for (int i = 0; i <= nx; i++) {
			int node_id = j * (nx + 1) + i;
			nodes[node_id].x = i * lx / nx;
			nodes[node_id].y = j * ly / ny;
		}
	}

	elements.resize(nx * ny);
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			int elem_id = j * nx + i;
			Element& elem = elements[elem_id];
			elem.nodes[0] = j * (nx + 1) + i;
			elem.nodes[1] = j * (nx + 1) + i + 1;
			elem.nodes[2] = (j + 1) * (nx + 1) + i + 1;
			elem.nodes[3] = (j + 1) * (nx + 1) + i;
			for (int k = 0; k < NODES_PER_ELEMENT; k++) {
				int node_id = elem.nodes[k];
				elem.x[k] = nodes[node_id].x;
				elem.y[k] = nodes[node_id].y;
			}
			elem.color = -1; // 初始化颜色为-1
		}
	}
}

void buildCSRStructure(int num_nodes, int num_elements,
	const std::vector<Element>& elements,
	std::vector<int>& row_ptr,
	std::vector<int>& col_ind) {
	int total_dofs = num_nodes * DOF_PER_NODE;

	// 第一步：计算每行的非零元数量
	std::vector<std::set<int>> column_indices(total_dofs);

	for (const auto& elem : elements) {
		int dof_indices[ELEM_DOF];
		for (int i = 0; i < NODES_PER_ELEMENT; i++) {
			int node_id = elem.nodes[i];
			dof_indices[i * 2] = node_id * 2;
			dof_indices[i * 2 + 1] = node_id * 2 + 1;
		}

		for (int i = 0; i < ELEM_DOF; i++) {
			int row = dof_indices[i];
			for (int j = 0; j < ELEM_DOF; j++) {
				int col = dof_indices[j];
				column_indices[row].insert(col);
			}
		}
	}

	// 第二步：构建row_ptr
	row_ptr.resize(total_dofs + 1);
	row_ptr[0] = 0;
	for (int i = 0; i < total_dofs; i++) {
		row_ptr[i + 1] = row_ptr[i] + column_indices[i].size();
	}

	// 第三步：构建col_ind
	int nnz = row_ptr[total_dofs];
	col_ind.resize(nnz);
	for (int i = 0; i < total_dofs; i++) {
		std::copy(column_indices[i].begin(), column_indices[i].end(),
			col_ind.begin() + row_ptr[i]);
	}
}

// ==================== 染色相关函数 ====================
int colorElements(std::vector<Element>& elements, std::vector<std::vector<int>>& color_groups) {
	int num_elements = elements.size();
	std::vector<std::set<int>> adjacents(num_elements);

	// 构建节点到单元的映射关系
	std::map<int, std::vector<int>> node_to_elements;
	for (int eid = 0; eid < num_elements; eid++) {
		for (int i = 0; i < NODES_PER_ELEMENT; i++) {
			int node_id = elements[eid].nodes[i];
			node_to_elements[node_id].push_back(eid);
		}
	}

	// 建立相邻关系
	for (int eid = 0; eid < num_elements; eid++) {
		for (int i = 0; i < NODES_PER_ELEMENT; i++) {
			int node_id = elements[eid].nodes[i];
			for (int adjacent_eid : node_to_elements[node_id]) {
				if (adjacent_eid != eid) {
					adjacents[eid].insert(adjacent_eid);
					adjacents[adjacent_eid].insert(eid);
				}
			}
		}
	}

	// 按度排序（从大到小）
	std::vector<int> degrees(num_elements, 0);
	for (int eid = 0; eid < num_elements; eid++) {
		degrees[eid] = adjacents[eid].size();
	}

	std::vector<int> indices(num_elements);
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&degrees](int a, int b) {
		return degrees[a] > degrees[b];
	});

	// 染色算法
	std::vector<int> colors(num_elements, -1);
	int max_color = 0;

	for (int idx = 0; idx < num_elements; idx++) {
		int eid = indices[idx];
		std::vector<bool> available(num_elements, true);

		// 检查所有相邻单元的颜色
		for (int adjacent : adjacents[eid]) {
			if (colors[adjacent] != -1) {
				available[colors[adjacent]] = false;
			}
		}

		// 分配最小可用颜色
		int color;
		for (color = 0; color < num_elements; color++) {
			if (available[color]) break;
		}

		colors[eid] = color;
		max_color = std::max(max_color, color);
		elements[eid].color = color;
	}

	int num_colors = max_color + 1;

	// 按颜色分组
	color_groups.resize(num_colors);
	for (int eid = 0; eid < num_elements; eid++) {
		color_groups[colors[eid]].push_back(eid);
	}

	return num_colors;
}

// ==================== 性能测试函数 ====================
PerformanceMetrics compareAssemblyMethods(
	int num_elements, int total_dofs, int nnz,
	const Element* d_elements,
	const int* d_csr_row_ptr,
	const int* d_csr_col_ind,
	double* d_csr_values_alls,
	double* d_csr_values_colored,
	const int* d_color_offsets,
	const int* d_color_sizes,
	int num_colors,
	const std::vector<int>& h_color_sizes) {

	PerformanceMetrics metrics;
	metrics.num_colors = num_colors;

	// 计算最大颜色组大小
	metrics.max_color_size = 0;
	for (int i = 0; i < num_colors; i++) {
		if (h_color_sizes[i] > metrics.max_color_size) {
			metrics.max_color_size = h_color_sizes[i];
		}
	}

	// 分配原子操作计数器
	int* d_global_atomic_counter_alls;
	CHECK_CUDA_ERROR(cudaMalloc(&d_global_atomic_counter_alls, sizeof(int)));

	// 设置网格和块大小
	int threads = BLOCK_SIZE;
	int blocks_alls = (num_elements + threads - 1) / threads;
	int blocks_convert = (total_dofs + threads - 1) / threads;

	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	// ==========================================
	// A-LLS 预分配与初始化
	// ==========================================
	size_t free_mem_start, total_mem;
	CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem_start, &total_mem));
	size_t mem_start_alls = free_mem_start;

	// 分配A-LLS数据结构
	ALLSNode** d_alls_row_heads;
	ALLSNode* h_d_alls_pool;
	int* h_d_alls_pool_index;

	CHECK_CUDA_ERROR(cudaMalloc(&d_alls_row_heads, total_dofs * sizeof(ALLSNode*)));
	
	// 估计A-LLS节点池大小
	int alls_pool_size = num_elements * ELEM_DOF * ELEM_DOF;
	CHECK_CUDA_ERROR(cudaMalloc(&h_d_alls_pool, alls_pool_size * sizeof(ALLSNode)));
	CHECK_CUDA_ERROR(cudaMalloc(&h_d_alls_pool_index, sizeof(int)));

	// 设置设备全局变量
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool, &h_d_alls_pool, sizeof(ALLSNode*)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool_index, &h_d_alls_pool_index, sizeof(int*)));

	// --- A-LLS 预热迭代 ---
	for (int i = 0; i < WARMUP_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_global_atomic_counter_alls, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_csr_values_alls, 0, nnz * sizeof(double)));
		CHECK_CUDA_ERROR(cudaMemset(d_alls_row_heads, 0, total_dofs * sizeof(ALLSNode*)));
		CHECK_CUDA_ERROR(cudaMemset(h_d_alls_pool_index, 0, sizeof(int))); // 重要：重置内存池

		assembleGlobalStiffnessKernelALLS << <blocks_alls, threads >> > (
			num_elements, d_elements, d_alls_row_heads, d_global_atomic_counter_alls);
		convertALLStoCSRKernel << <blocks_convert, threads >> > (
			total_dofs, d_alls_row_heads, d_csr_values_alls, d_csr_row_ptr, d_csr_col_ind);
	}
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	// --- A-LLS 性能测量 ---
	float total_alls_ms = 0.0f;
	for (int i = 0; i < MEASURE_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_global_atomic_counter_alls, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_csr_values_alls, 0, nnz * sizeof(double)));
		CHECK_CUDA_ERROR(cudaMemset(d_alls_row_heads, 0, total_dofs * sizeof(ALLSNode*)));
		CHECK_CUDA_ERROR(cudaMemset(h_d_alls_pool_index, 0, sizeof(int))); // 重要：重置内存池

		CHECK_CUDA_ERROR(cudaEventRecord(start));
		assembleGlobalStiffnessKernelALLS << <blocks_alls, threads >> > (
			num_elements, d_elements, d_alls_row_heads, d_global_atomic_counter_alls);
		convertALLStoCSRKernel << <blocks_convert, threads >> > (
			total_dofs, d_alls_row_heads, d_csr_values_alls, d_csr_row_ptr, d_csr_col_ind);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		float iter_ms = 0.0f;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&iter_ms, start, stop));
		total_alls_ms += iter_ms;
	}
	metrics.alls_time_ms = total_alls_ms / MEASURE_ITERS;

	// 计算A-LLS方法内存使用量
	size_t free_mem_end_alls;
	CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem_end_alls, &total_mem));
	metrics.alls_mem_usage = mem_start_alls - free_mem_end_alls;

	int h_global_atomic_counter_alls;
	CHECK_CUDA_ERROR(cudaMemcpy(&h_global_atomic_counter_alls, d_global_atomic_counter_alls,
		sizeof(int), cudaMemcpyDeviceToHost));
	metrics.atomic_ops_alls = h_global_atomic_counter_alls;

	// 估算A-LLS方法的浮点运算性能
	metrics.alls_flops = num_elements * 4 * (1000 + 100 + 200) / 1e9; // 简化估算
	metrics.alls_flops /= (metrics.alls_time_ms / 1000.0f); // 转换为GFLOPS

	// 清理A-LLS数据结构
	CHECK_CUDA_ERROR(cudaFree(d_alls_row_heads));
	CHECK_CUDA_ERROR(cudaFree(h_d_alls_pool));
	CHECK_CUDA_ERROR(cudaFree(h_d_alls_pool_index));


	// ==========================================
	// 染色法 性能测试（无原子操作版本）
	// ==========================================
	size_t mem_start_colored;
	CHECK_CUDA_ERROR(cudaMemGetInfo(&mem_start_colored, &total_mem));

	// 动态调整线程块大小
	dim3 blocks_colored_noatomic(num_colors);
	int threads_per_block = std::min(metrics.max_color_size, 256);
	if (threads_per_block < 32) threads_per_block = 32;
	dim3 threads_colored_noatomic(threads_per_block);

	// --- 染色法 预热迭代 ---
	for (int i = 0; i < WARMUP_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_csr_values_colored, 0, nnz * sizeof(double)));
		assembleGlobalStiffnessKernelColoredNoAtomic << <blocks_colored_noatomic, threads_colored_noatomic >> > (
			num_elements, d_elements, d_csr_values_colored,
			d_csr_row_ptr, d_csr_col_ind,
			d_color_offsets, d_color_sizes, num_colors, total_dofs, nnz);
	}
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	// --- 染色法 性能测量 ---
	float total_colored_ms = 0.0f;
	for (int i = 0; i < MEASURE_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_csr_values_colored, 0, nnz * sizeof(double)));
		
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		assembleGlobalStiffnessKernelColoredNoAtomic << <blocks_colored_noatomic, threads_colored_noatomic >> > (
			num_elements, d_elements, d_csr_values_colored,
			d_csr_row_ptr, d_csr_col_ind,
			d_color_offsets, d_color_sizes, num_colors, total_dofs, nnz);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		float iter_ms = 0.0f;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&iter_ms, start, stop));
		total_colored_ms += iter_ms;
	}
	metrics.colored_time_ms = total_colored_ms / MEASURE_ITERS;

	// 计算染色方法内存使用量
	size_t free_mem_end_colored;
	CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem_end_colored, &total_mem));
	metrics.colored_mem_usage = mem_start_colored - free_mem_end_colored;

	// 计算刚度矩阵内存大小
	metrics.stiffness_matrix_size = nnz * sizeof(double);

	// 估算染色方法的浮点运算性能
	metrics.colored_flops = num_elements * 4 * (1000 + 100 + 200) / 1e9; // 简化估算
	metrics.colored_flops /= (metrics.colored_time_ms / 1000.0f); // 转换为GFLOPS

	// 计算性能指标
	metrics.speedup = metrics.colored_time_ms / metrics.alls_time_ms;

	// 清理
	CHECK_CUDA_ERROR(cudaFree(d_global_atomic_counter_alls));
	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));

	return metrics;
}

// ==================== 运行单个测试的函数 ====================
TestConfig runSingleTest(int nx, int ny) {
	TestConfig config;
	config.nx = nx;
	config.ny = ny;
	config.num_elements = nx * ny;

	double lx = 1.0;
	double ly = 1.0;

	// 生成网格
	std::vector<Node> nodes;
	std::vector<Element> elements;
	generateMesh(nx, ny, lx, ly, nodes, elements);

	// 构建CSR结构
	std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	int num_nodes = (nx + 1) * (ny + 1);
	buildCSRStructure(num_nodes, config.num_elements, elements, csr_row_ptr, csr_col_ind);
	int total_dofs = num_nodes * DOF_PER_NODE;
	int nnz = csr_row_ptr[total_dofs];

	std::cout << "CSR structure: " << nnz << " non-zero entries" << std::endl;

	// 测量染色时间
	auto start_color = std::chrono::high_resolution_clock::now();

	// 对单元进行染色
	std::vector<std::vector<int>> color_groups;
	int num_colors = colorElements(elements, color_groups);

	auto stop_color = std::chrono::high_resolution_clock::now();
	auto duration_color = std::chrono::duration_cast<std::chrono::microseconds>(stop_color - start_color);
	float coloring_time_ms = duration_color.count() / 1000.0f;

	std::cout << "Number of colors: " << num_colors << std::endl;

	// 准备颜色偏移和大小数组
	std::vector<int> color_offsets(num_colors);
	std::vector<int> color_sizes(num_colors);
	int offset = 0;
	for (int i = 0; i < num_colors; i++) {
		color_offsets[i] = offset;
		color_sizes[i] = color_groups[i].size();
		offset += color_sizes[i];
	}

	// 分配设备内存
	Element* d_elements = nullptr;
	int* d_csr_row_ptr = nullptr;
	int* d_csr_col_ind = nullptr;
	double* d_csr_values_alls = nullptr;
	double* d_csr_values_colored = nullptr;
	int* d_color_offsets = nullptr;
	int* d_color_sizes = nullptr;

	CHECK_CUDA_ERROR(cudaMalloc(&d_elements, config.num_elements * sizeof(Element)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_row_ptr, (total_dofs + 1) * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_values_alls, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_values_colored, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_color_offsets, num_colors * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_color_sizes, num_colors * sizeof(int)));

	// 拷贝数据到设备
	CHECK_CUDA_ERROR(cudaMemcpy(d_elements, elements.data(),
		config.num_elements * sizeof(Element), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(),
		(total_dofs + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(),
		nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_color_offsets, color_offsets.data(),
		num_colors * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_color_sizes, color_sizes.data(),
		num_colors * sizeof(int), cudaMemcpyHostToDevice));

	// 比较两种方法的性能
	PerformanceMetrics metrics = compareAssemblyMethods(
		config.num_elements, total_dofs, nnz,
		d_elements,
		d_csr_row_ptr, d_csr_col_ind,
		d_csr_values_alls, d_csr_values_colored,
		d_color_offsets, d_color_sizes, num_colors,
		color_sizes); // 传入颜色大小信息

	// 保存染色时间
	metrics.coloring_time_ms = coloring_time_ms;

	// 估算原子操作时间（假设每个原子操作耗时约100纳秒）
	metrics.atomic_time_ms = metrics.atomic_ops_alls * 0.0001f; // 每个原子操作0.1微秒

   // 确保原子操作时间不超过总时间的90%
	if (metrics.atomic_time_ms > metrics.alls_time_ms * 0.9f) {
		metrics.atomic_time_ms = metrics.alls_time_ms * 0.9f;
	}

	// 将metrics赋值给config.metrics
	config.metrics = metrics;

	// 清理设备内存
	CHECK_CUDA_ERROR(cudaFree(d_elements));
	CHECK_CUDA_ERROR(cudaFree(d_csr_row_ptr));
	CHECK_CUDA_ERROR(cudaFree(d_csr_col_ind));
	CHECK_CUDA_ERROR(cudaFree(d_csr_values_alls));
	CHECK_CUDA_ERROR(cudaFree(d_csr_values_colored));
	CHECK_CUDA_ERROR(cudaFree(d_color_offsets));
	CHECK_CUDA_ERROR(cudaFree(d_color_sizes));

	return config;
}

// ==================== 主函数 ====================
int main() {
	std::cout << "Starting FEM assembly performance comparison (A-LLS vs Colored)..." << std::endl;
	std::cout << "Warm-up iterations: " << WARMUP_ITERS << ", Measurement iterations: " << MEASURE_ITERS << std::endl;

	// 定义要测试的网格规模
	std::vector<std::pair<int, int>> test_cases = {
		{10, 10},    // 100 elements
		{100, 10},    // ~1000 elements (1000)
		{100, 100},  // 10,000 elements
	};

	std::vector<TestConfig> results;

	// 运行所有测试用例
	for (const auto& test_case : test_cases) {
		int nx = test_case.first;
		int ny = test_case.second;
		int num_elements = nx * ny;

		std::cout << "\n==============================================" << std::endl;
		std::cout << "Running test for " << num_elements << " elements (" << nx << "x" << ny << ")" << std::endl;
		std::cout << "==============================================" << std::endl;

		TestConfig config = runSingleTest(nx, ny);
		results.push_back(config);
	}

	// 输出性能对比结果 - 计算时间和效率部分
	std::cout << "\n\n========== Performance Comparison Summary (Time & Efficiency) ==========" << std::endl;
	std::cout << std::setw(12) << "Elements"
		<< std::setw(15) << "A-LLS Time(ms)"
		<< std::setw(15) << "Colored Time(ms)"
		<< std::setw(15) << "Speedup"
		<< std::setw(15) << "Atomic Ops"
		<< std::endl;

	for (const auto& result : results) {
		std::cout << std::setw(12) << result.num_elements
			<< std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.alls_time_ms
			<< std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.colored_time_ms
			<< std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.speedup
			<< std::setw(15) << result.metrics.atomic_ops_alls
			<< std::endl;
	}

	// 输出性能对比结果 - 内存使用部分
	std::cout << "\n\n========== Performance Comparison Summary (Memory Usage) ==========" << std::endl;
	std::cout << std::setw(12) << "Elements"
		<< std::setw(15) << "A-LLS Mem(MB)"
		<< std::setw(15) << "Colored Mem(MB)"
		<< std::setw(15) << "Matrix Size(MB)"
		<< std::setw(15) << "Total Mem(MB)"
		<< std::endl;

	for (const auto& result : results) {
		double alls_mem_mb = result.metrics.alls_mem_usage / (1024.0 * 1024.0);
		double colored_mem_mb = result.metrics.colored_mem_usage / (1024.0 * 1024.0);
		double matrix_size_mb = result.metrics.stiffness_matrix_size / (1024.0 * 1024.0);
		double total_mem_mb = alls_mem_mb + colored_mem_mb + matrix_size_mb;

		std::cout << std::setw(12) << result.num_elements
			<< std::setw(15) << std::fixed << std::setprecision(2) << alls_mem_mb
			<< std::setw(15) << std::fixed << std::setprecision(2) << colored_mem_mb
			<< std::setw(15) << std::fixed << std::setprecision(2) << matrix_size_mb
			<< std::setw(15) << std::fixed << std::setprecision(2) << total_mem_mb
			<< std::endl;
	}

	// 输出浮点运算性能部分
	std::cout << "\n\n========== Floating Point Performance (GFLOPS) ==========" << std::endl;
	std::cout << std::setw(12) << "Elements"
		<< std::setw(15) << "A-LLS GFLOPS"
		<< std::setw(15) << "Colored GFLOPS"
		<< std::setw(15) << "Performance Ratio"
		<< std::endl;

	for (const auto& result : results) {
		double performance_ratio = result.metrics.alls_flops / result.metrics.colored_flops;

		std::cout << std::setw(12) << result.num_elements
			<< std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.alls_flops
			<< std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.colored_flops
			<< std::setw(15) << std::fixed << std::setprecision(2) << performance_ratio
			<< std::endl;
	}

	// 输出染色时间表格
	std::cout << "\n\n========== Host-Side Coloring Time ==========" << std::endl;
	std::cout << std::setw(12) << "Elements"
		<< std::setw(20) << "Coloring Time(ms)"
		<< std::endl;

	for (const auto& result : results) {
		std::cout << std::setw(12) << result.num_elements
			<< std::setw(20) << std::fixed << std::setprecision(2) << result.metrics.coloring_time_ms
			<< std::endl;
	}

	// 输出原子操作信息表格
	std::cout << "\n\n========== Atomic Operations in A-LLS ==========" << std::endl;
	std::cout << std::setw(12) << "Elements"
		<< std::setw(20) << "Atomic Ops"
		<< std::setw(20) << "Atomic Time(ms)"
		<< std::setw(20) << "Fraction of Total Time(%)"
		<< std::endl;

	for (const auto& result : results) {
		float fraction = (result.metrics.atomic_time_ms / result.metrics.alls_time_ms) * 100.0f;

		std::cout << std::setw(12) << result.num_elements
			<< std::setw(20) << result.metrics.atomic_ops_alls
			<< std::setw(20) << std::fixed << std::setprecision(2) << result.metrics.atomic_time_ms
			<< std::setw(20) << std::fixed << std::setprecision(1) << fraction
			<< std::endl;
	}

	// 将结果保存到CSV文件以便进一步分析
	std::ofstream perf_file("performance_results.csv");
	perf_file << "Elements,ALLS_Time,Colored_Time,Speedup,Atomic_Ops,"
		<< "ALLS_Mem_MB,Colored_Mem_MB,Matrix_Size_MB,"
		<< "Num_Colors,Max_Color_Size,ALLS_GFLOPS,Colored_GFLOPS\n";

	for (const auto& result : results) {
		perf_file << result.num_elements << ","
			<< result.metrics.alls_time_ms << ","
			<< result.metrics.colored_time_ms << ","
			<< result.metrics.speedup << ","
			<< result.metrics.atomic_ops_alls << ","
			<< result.metrics.alls_mem_usage / (1024.0 * 1024.0) << ","
			<< result.metrics.colored_mem_usage / (1024.0 * 1024.0) << ","
			<< result.metrics.stiffness_matrix_size / (1024.0 * 1024.0) << ","
			<< result.metrics.num_colors << ","
			<< result.metrics.max_color_size << ","
			<< result.metrics.alls_flops << ","
			<< result.metrics.colored_flops << "\n";
	}
	perf_file.close();

	std::cout << "\nResults saved to performance_results.csv" << std::endl;
	std::cout << "Program finished successfully!" << std::endl;
	return 0;
}
