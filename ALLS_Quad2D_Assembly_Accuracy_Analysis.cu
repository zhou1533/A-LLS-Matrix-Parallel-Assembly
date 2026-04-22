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

// 精度分析结果结构体
struct PrecisionAnalysis {
	double max_abs_error_alls;
	double max_rel_error_alls;
	double mean_abs_error_alls;
	double mean_rel_error_alls;
	double frobenius_norm_error_alls;

	double max_abs_error_colored;
	double max_rel_error_colored;
	double mean_abs_error_colored;
	double mean_rel_error_colored;
	double frobenius_norm_error_colored;
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

// ==================== 性能分析结构体 ====================
struct PerformanceMetrics {
	float alls_time_ms;
	float colored_time_ms;
	float speedup; // 染色方法相对于A-LLS的加速比
	int atomic_ops_alls;
};

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

	// 首先清零CSR行
	for (int i = row_start; i <= row_end; i++) {
		csr_values[i] = 0.0;
	}

	// 然后累加A-LLS中的值
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

	// 严格验证染色结果
	bool has_conflict = false;
	for (int eid = 0; eid < num_elements; eid++) {
		for (int adjacent : adjacents[eid]) {
			if (colors[eid] == colors[adjacent]) {
				std::cerr << "染色冲突: 单元 " << eid << " 和 " << adjacent
					<< " 有相同颜色 " << colors[eid] << std::endl;
				has_conflict = true;

				// 立即修复冲突
				int new_color = num_colors++;
				colors[eid] = new_color;
				color_groups.resize(num_colors);
				color_groups[new_color].push_back(eid);
				elements[eid].color = new_color;
			}
		}
	}

	if (has_conflict) {
		std::cerr << "发现染色冲突，已尝试修复。新颜色数量: " << num_colors << std::endl;
	}
	else {
		std::cout << "染色验证通过，无冲突" << std::endl;
	}

	return num_colors;
}

// ==================== 性能测试函数 ====================
PerformanceMetrics compareAssemblyMethods(int num_elements, int total_dofs, int nnz,
	const Element* d_elements,
	const int* d_csr_row_ptr,
	const int* d_csr_col_ind,
	double* d_csr_values_alls,
	double* d_csr_values_colored,
	const int* d_color_offsets,
	const int* d_color_sizes,
	int num_colors) {

	PerformanceMetrics metrics;

	// 分配原子操作计数器
	int* d_global_atomic_counter_alls;
	CHECK_CUDA_ERROR(cudaMalloc(&d_global_atomic_counter_alls, sizeof(int)));

	// 设置网格和块大小
	int threads = BLOCK_SIZE;
	int blocks_alls = (num_elements + threads - 1) / threads;

	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	// 测试A-LLS方法
	CHECK_CUDA_ERROR(cudaMemset(d_global_atomic_counter_alls, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(d_csr_values_alls, 0, nnz * sizeof(double)));

	// 分配A-LLS数据结构
	ALLSNode** d_alls_row_heads;
	ALLSNode* h_d_alls_pool;
	int* h_d_alls_pool_index;

	CHECK_CUDA_ERROR(cudaMalloc(&d_alls_row_heads, total_dofs * sizeof(ALLSNode*)));
	CHECK_CUDA_ERROR(cudaMemset(d_alls_row_heads, 0, total_dofs * sizeof(ALLSNode*)));

	// 估计A-LLS节点池大小
	int alls_pool_size = num_elements * ELEM_DOF * ELEM_DOF;
	CHECK_CUDA_ERROR(cudaMalloc(&h_d_alls_pool, alls_pool_size * sizeof(ALLSNode)));
	CHECK_CUDA_ERROR(cudaMalloc(&h_d_alls_pool_index, sizeof(int)));

	// 初始化A-LLS节点池索引
	int init_alls_pool_index = 0;
	CHECK_CUDA_ERROR(cudaMemcpy(h_d_alls_pool_index, &init_alls_pool_index, sizeof(int), cudaMemcpyHostToDevice));

	// 设置设备全局变量
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool, &h_d_alls_pool, sizeof(ALLSNode*)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool_index, &h_d_alls_pool_index, sizeof(int*)));

	CHECK_CUDA_ERROR(cudaEventRecord(start));

	// 使用A-LLS数据结构组装
	assembleGlobalStiffnessKernelALLS << <blocks_alls, threads >> > (
		num_elements, d_elements, d_alls_row_heads, d_global_atomic_counter_alls);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	// 将A-LLS转换为CSR格式
	int blocks_convert = (total_dofs + threads - 1) / threads;
	convertALLStoCSRKernel << <blocks_convert, threads >> > (
		total_dofs, d_alls_row_heads, d_csr_values_alls,
		d_csr_row_ptr, d_csr_col_ind);

	CHECK_CUDA_ERROR(cudaEventRecord(stop));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.alls_time_ms, start, stop));

	int h_global_atomic_counter_alls;
	CHECK_CUDA_ERROR(cudaMemcpy(&h_global_atomic_counter_alls, d_global_atomic_counter_alls,
		sizeof(int), cudaMemcpyDeviceToHost));
	metrics.atomic_ops_alls = h_global_atomic_counter_alls;

	// 清理A-LLS数据结构
	CHECK_CUDA_ERROR(cudaFree(d_alls_row_heads));
	CHECK_CUDA_ERROR(cudaFree(h_d_alls_pool));
	CHECK_CUDA_ERROR(cudaFree(h_d_alls_pool_index));

	// 测试染色方法（无原子操作版本）
	CHECK_CUDA_ERROR(cudaMemset(d_csr_values_colored, 0, nnz * sizeof(double)));

	CHECK_CUDA_ERROR(cudaEventRecord(start));

	// 使用染色方法组装（无原子操作）
	// 动态调整线程块大小
	dim3 blocks_colored_noatomic(num_colors);

	// 将颜色大小从设备拷贝到主机
	std::vector<int> h_color_sizes(num_colors);
	CHECK_CUDA_ERROR(cudaMemcpy(h_color_sizes.data(), d_color_sizes,
		num_colors * sizeof(int), cudaMemcpyDeviceToHost));

	// 计算最大颜色组大小
	int max_color_size = 0;
	for (int i = 0; i < num_colors; i++) {
		if (h_color_sizes[i] > max_color_size) {
			max_color_size = h_color_sizes[i];
		}
	}

	// 动态调整线程块大小
	int threads_per_block = min(max_color_size, 256);
	if (threads_per_block < 32) {
		threads_per_block = 32; // 确保至少有32个线程
	}

	dim3 threads_colored_noatomic(threads_per_block);

	assembleGlobalStiffnessKernelColoredNoAtomic << <blocks_colored_noatomic, threads_colored_noatomic >> > (
		num_elements, d_elements, d_csr_values_colored,
		d_csr_row_ptr, d_csr_col_ind,
		d_color_offsets, d_color_sizes, num_colors, total_dofs, nnz);

	CHECK_CUDA_ERROR(cudaEventRecord(stop));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaEventElapsedTime(&metrics.colored_time_ms, start, stop));

	// 计算性能指标
	metrics.speedup = metrics.colored_time_ms / metrics.alls_time_ms;

	// 清理
	CHECK_CUDA_ERROR(cudaFree(d_global_atomic_counter_alls));
	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));

	return metrics;
}

// ==================== 精度分析函数 ====================
PrecisionAnalysis analyzePrecision(int nnz, const std::vector<double>& host_reference,
	const std::vector<double>& alls_result,
	const std::vector<double>& colored_result) {
	PrecisionAnalysis analysis;

	// 初始化误差指标
	analysis.max_abs_error_alls = 0.0;
	analysis.max_rel_error_alls = 0.0;
	analysis.mean_abs_error_alls = 0.0;
	analysis.mean_rel_error_alls = 0.0;
	analysis.frobenius_norm_error_alls = 0.0;

	analysis.max_abs_error_colored = 0.0;
	analysis.max_rel_error_colored = 0.0;
	analysis.mean_abs_error_colored = 0.0;
	analysis.mean_rel_error_colored = 0.0;
	analysis.frobenius_norm_error_colored = 0.0;

	// 计算参考矩阵的Frobenius范数
	double host_frobenius_norm = 0.0;
	for (int i = 0; i < nnz; i++) {
		host_frobenius_norm += host_reference[i] * host_reference[i];
	}
	host_frobenius_norm = sqrt(host_frobenius_norm);

	// 分析A-LLS方法的精度
	for (int i = 0; i < nnz; i++) {
		if (host_reference[i] != 0.0) { // 避免除以零
			double abs_error = fabs(alls_result[i] - host_reference[i]);
			double rel_error = abs_error / fabs(host_reference[i]);

			analysis.max_abs_error_alls = std::max(analysis.max_abs_error_alls, abs_error);
			analysis.max_rel_error_alls = std::max(analysis.max_rel_error_alls, rel_error);
			analysis.mean_abs_error_alls += abs_error;
			analysis.mean_rel_error_alls += rel_error;
			analysis.frobenius_norm_error_alls += (alls_result[i] - host_reference[i]) *
				(alls_result[i] - host_reference[i]);
		}
	}

	analysis.mean_abs_error_alls /= nnz;
	analysis.mean_rel_error_alls /= nnz;
	analysis.frobenius_norm_error_alls = sqrt(analysis.frobenius_norm_error_alls) / host_frobenius_norm;

	// 分析染色方法的精度
	for (int i = 0; i < nnz; i++) {
		if (host_reference[i] != 0.0) { // 避免除以零
			double abs_error = fabs(colored_result[i] - host_reference[i]);
			double rel_error = abs_error / fabs(host_reference[i]);

			analysis.max_abs_error_colored = std::max(analysis.max_abs_error_colored, abs_error);
			analysis.max_rel_error_colored = std::max(analysis.max_rel_error_colored, rel_error);
			analysis.mean_abs_error_colored += abs_error;
			analysis.mean_rel_error_colored += rel_error;
			analysis.frobenius_norm_error_colored += (colored_result[i] - host_reference[i]) *
				(colored_result[i] - host_reference[i]);
		}
	}

	analysis.mean_abs_error_colored /= nnz;
	analysis.mean_rel_error_colored /= nnz;
	analysis.frobenius_norm_error_colored = sqrt(analysis.frobenius_norm_error_colored) / host_frobenius_norm;

	return analysis;
}

// ==================== 主函数 ====================
int main() {
	std::cout << "Starting FEM assembly performance comparison (A-LLS vs Colored)..." << std::endl;

	// 设置网格规模
	int nx = 10;
	int ny = 100;
	double lx = 1.0;
	double ly = 1.0;

	int num_nodes = (nx + 1) * (ny + 1);
	int num_elements = nx * ny;
	int total_dofs = num_nodes * DOF_PER_NODE;

	std::cout << "Mesh: " << nx << "x" << ny << " elements, "
		<< num_nodes << " nodes, " << total_dofs << " DOFs" << std::endl;

	// 生成网格
	std::vector<Node> nodes;
	std::vector<Element> elements;
	generateMesh(nx, ny, lx, ly, nodes, elements);

	// 构建CSR结构
	std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	buildCSRStructure(num_nodes, num_elements, elements, csr_row_ptr, csr_col_ind);
	int nnz = csr_row_ptr[total_dofs];

	std::cout << "CSR structure: " << nnz << " non-zero entries" << std::endl;

	// 对单元进行染色
	std::vector<std::vector<int>> color_groups;
	int num_colors = colorElements(elements, color_groups);

	std::cout << "Number of colors: " << num_colors << std::endl;
	for (int i = 0; i < num_colors; i++) {
		std::cout << "Color " << i << ": " << color_groups[i].size() << " elements" << std::endl;
	}

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

	CHECK_CUDA_ERROR(cudaMalloc(&d_elements, num_elements * sizeof(Element)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_row_ptr, (total_dofs + 1) * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_values_alls, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_values_colored, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_color_offsets, num_colors * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_color_sizes, num_colors * sizeof(int)));

	// 拷贝数据到设备
	CHECK_CUDA_ERROR(cudaMemcpy(d_elements, elements.data(),
		num_elements * sizeof(Element), cudaMemcpyHostToDevice));
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
		num_elements, total_dofs, nnz,
		d_elements,
		d_csr_row_ptr, d_csr_col_ind,
		d_csr_values_alls, d_csr_values_colored,
		d_color_offsets, d_color_sizes, num_colors);

	// 输出性能结果
	std::cout << "\n========== Performance Comparison ==========" << std::endl;
	std::cout << "A-LLS method time: " << metrics.alls_time_ms << " ms" << std::endl;
	std::cout << "Colored method time: " << metrics.colored_time_ms << " ms" << std::endl;
	std::cout << " A-LLS method speedup (vs Colored method): " << metrics.speedup << "x" << std::endl;
	std::cout << "Atomic operations (A-LLS): " << metrics.atomic_ops_alls << std::endl;

	// 获取GPU计算结果
	std::vector<double> h_csr_values_alls(nnz);
	std::vector<double> h_csr_values_colored(nnz);

	CHECK_CUDA_ERROR(cudaMemcpy(h_csr_values_alls.data(), d_csr_values_alls,
		nnz * sizeof(double), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(h_csr_values_colored.data(), d_csr_values_colored,
		nnz * sizeof(double), cudaMemcpyDeviceToHost));

	// ==================== 主机端计算真实刚度矩阵 ====================
	std::cout << "\n========== Host Assembly ==========" << std::endl;

	std::vector<double> h_csr_values_host(nnz);
	assembleGlobalStiffnessHost(num_elements, elements, csr_row_ptr, csr_col_ind, h_csr_values_host);

	// 进行精度分析
	PrecisionAnalysis precision = analyzePrecision(nnz, h_csr_values_host,
		h_csr_values_alls, h_csr_values_colored);

	// 输出精度分析结果
	std::cout << "\n========== Precision Analysis ==========" << std::endl;
	std::cout << "A-LLS Method Precision:" << std::endl;
	std::cout << "  Max Absolute Error: " << precision.max_abs_error_alls << std::endl;
	std::cout << "  Max Relative Error: " << precision.max_rel_error_alls << std::endl;
	std::cout << "  Mean Absolute Error: " << precision.mean_abs_error_alls << std::endl;
	std::cout << "  Mean Relative Error: " << precision.mean_rel_error_alls << std::endl;
	std::cout << "  Frobenius Norm Error: " << precision.frobenius_norm_error_alls << std::endl;

	std::cout << "\nColored Method Precision:" << std::endl;
	std::cout << "  Max Absolute Error: " << precision.max_abs_error_colored << std::endl;
	std::cout << "  Max Relative Error: " << precision.max_rel_error_colored << std::endl;
	std::cout << "  Mean Absolute Error: " << precision.mean_abs_error_colored << std::endl;
	std::cout << "  Mean Relative Error: " << precision.mean_rel_error_colored << std::endl;
	std::cout << "  Frobenius Norm Error: " << precision.frobenius_norm_error_colored << std::endl;

	// 提取前十行十列的非零元
	int rows_to_show = total_dofs;
	int cols_to_show = total_dofs;

	std::cout << "\nTop 10x10 elements of the global stiffness matrix (host computation):" << std::endl;
	std::cout << std::fixed << std::setprecision(6);

	for (int i = 0; i < 10; i++) {
		int row_start = csr_row_ptr[i];
		int row_end = csr_row_ptr[i + 1] - 1;

		std::cout << "Row " << i << ": ";
		for (int j = 0; j < 10; j++) {
			// 查找列j是否在非零元中
			bool found = false;
			double value = 0.0;

			for (int k = row_start; k <= row_end; k++) {
				if (csr_col_ind[k] == j) {
					found = true;
					value = h_csr_values_host[k];
					break;
				}
			}

			if (found) {
				std::cout << std::setw(12) << value << " ";
			}
			else {
				std::cout << std::setw(12) << "0.0" << " ";
			}
		}
		std::cout << std::endl;
	}

	// 比较主机结果与两种GPU方法
	std::cout << "\n========== Comparison with Host Results ==========" << std::endl;

	double max_diff_alls = 0.0;
	double max_diff_colored = 0.0;
	int error_count_alls = 0;
	int error_count_colored = 0;

	for (int i = 0; i < rows_to_show; i++) {
		int row_start = csr_row_ptr[i];
		int row_end = csr_row_ptr[i + 1] - 1;

		for (int j = 0; j < cols_to_show; j++) {
			// 查找列j是否在非零元中
			int pos = -1;
			for (int k = row_start; k <= row_end; k++) {
				if (csr_col_ind[k] == j) {
					pos = k;
					break;
				}
			}

			if (pos >= 0) {
				double host_val = h_csr_values_host[pos];
				double alls_val = h_csr_values_alls[pos];
				double colored_val = h_csr_values_colored[pos];

				double diff_alls = fabs(host_val - alls_val);
				double diff_colored = fabs(host_val - colored_val);

				if (diff_alls > 1e-10) {
					error_count_alls++;
					if (diff_alls > max_diff_alls) max_diff_alls = diff_alls;
				}

				if (diff_colored > 1e-10) {
					error_count_colored++;
					if (diff_colored > max_diff_colored) max_diff_colored = diff_colored;
				}
			}
		}
	}

	std::cout << "A-LLS method errors: " << error_count_alls << " values differ from host" << std::endl;
	std::cout << "Colored method errors: " << error_count_colored << " values differ from host" << std::endl;
	std::cout << "Max difference (A-LLS): " << max_diff_alls << std::endl;
	std::cout << "Max difference (Colored): " << max_diff_colored << std::endl;

	if (error_count_alls == 0 && error_count_colored == 0) {
		std::cout << "Both methods produce correct results compared to host!" << std::endl;
	}
	else if (error_count_alls == 0) {
		std::cout << "Only A-LLS method produces correct results!" << std::endl;
	}
	else if (error_count_colored == 0) {
		std::cout << "Only Colored method produces correct results!" << std::endl;
	}
	else {
		std::cout << "Both methods have errors compared to host!" << std::endl;
	}

	// 清理设备内存
	CHECK_CUDA_ERROR(cudaFree(d_elements));
	CHECK_CUDA_ERROR(cudaFree(d_csr_row_ptr));
	CHECK_CUDA_ERROR(cudaFree(d_csr_col_ind));
	CHECK_CUDA_ERROR(cudaFree(d_csr_values_alls));
	CHECK_CUDA_ERROR(cudaFree(d_csr_values_colored));
	CHECK_CUDA_ERROR(cudaFree(d_color_offsets));
	CHECK_CUDA_ERROR(cudaFree(d_color_sizes));

	std::cout << "\nProgram finished successfully!" << std::endl;
	return 0;
}
