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
#include <random>
#include <string>

// ==================== 错误检查宏 ====================
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
const int MAX_NODES_PER_ELEMENT = 4; // 最大节点数支持Q4
const int MAX_ELEM_DOF = MAX_NODES_PER_ELEMENT * DOF_PER_NODE;
const int BLOCK_SIZE = 256;

// 测试参数：预热与测量次数
const int WARMUP_ITERS = 3;
const int MEASURE_ITERS = 10;

// 单元类型枚举
enum ElementType {
	ELEM_Q4 = 0, // 4节点四边形 (Heavy Load: 4 Gauss Points)
	ELEM_T3 = 1  // 3节点三角形 (Light Load: 1 Gauss Point)
};

// ==================== 设备/主机共用常量 ====================
const int GAUSS_POINTS_Q4 = 4; // 2x2
const int GAUSS_POINTS_T3 = 1; // Centroid

__constant__ double d_YOUNG_MODULUS = 2.1e11;
__constant__ double d_POISSON_RATIO = 0.3;
__constant__ double d_THICKNESS = 0.01;

// Q4 高斯点 (2x2)
__constant__ double d_gauss_points_Q4[2] = { -0.577350269189626, 0.577350269189626 };
__constant__ double d_gauss_weights_Q4[2] = { 1.0, 1.0 };

const double h_YOUNG_MODULUS = 2.1e11;
const double h_POISSON_RATIO = 0.3;
const double h_THICKNESS = 0.01;

// ==================== 数据结构 ====================
struct Element {
	int nodes[MAX_NODES_PER_ELEMENT];
	double x[MAX_NODES_PER_ELEMENT];
	double y[MAX_NODES_PER_ELEMENT];
	int color;
	ElementType type;
	int num_nodes;
};

struct Node {
	double x, y;
};

// A-LLS 数据结构
struct ALLSNode {
	int col;
	double value;
	ALLSNode* next;
};

struct PerformanceMetrics {
	float alls_time_ms;
	float colored_time_ms;
	float speedup;
	int atomic_ops_alls;
	size_t alls_mem_usage;
	size_t colored_mem_usage;
	float coloring_time_ms;
};

// 新增：负载统计结构体
struct LoadStats {
	int count_q4;
	int count_t3;
	double percent_q4;
	double percent_t3;
	double avg_compute_intensity; // 基于积分点数的平均计算强度
	std::string load_description;
};

struct TestConfig {
	std::string name;
	int num_elements;
	int num_nodes;
	PerformanceMetrics metrics;
	LoadStats load_stats; // 包含负载统计
};

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

// ==================== 设备端计算逻辑 ====================

__device__ void computeDMatrix(double* D) {
	double E = d_YOUNG_MODULUS;
	double nu = d_POISSON_RATIO;
	double factor = E / (1.0 - nu * nu);
	D[0] = factor;         D[1] = factor * nu;   D[2] = 0.0;
	D[3] = factor * nu;    D[4] = factor;        D[5] = 0.0;
	D[6] = 0.0;            D[7] = 0.0;           D[8] = factor * (1.0 - nu) / 2.0;
}

__device__ void shapeFunctionsQ4(double xi, double eta, double* dNdxi, double* dNdeta) {
	dNdxi[0] = -0.25 * (1.0 - eta); dNdxi[1] = 0.25 * (1.0 - eta);
	dNdxi[2] = 0.25 * (1.0 + eta);  dNdxi[3] = -0.25 * (1.0 + eta);
	dNdeta[0] = -0.25 * (1.0 - xi); dNdeta[1] = -0.25 * (1.0 + xi);
	dNdeta[2] = 0.25 * (1.0 + xi);  dNdeta[3] = 0.25 * (1.0 - xi);
}

__device__ void shapeFunctionsT3(double* dNdxi, double* dNdeta) {
	dNdxi[0] = -1.0; dNdxi[1] = 1.0; dNdxi[2] = 0.0;
	dNdeta[0] = -1.0; dNdeta[1] = 0.0; dNdeta[2] = 1.0;
}

__device__ void computeElementStiffness(const Element& elem, double* ke) {
	for (int i = 0; i < MAX_ELEM_DOF * MAX_ELEM_DOF; i++) ke[i] = 0.0;

	double D[9];
	computeDMatrix(D);

	if (elem.type == ELEM_Q4) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double xi = d_gauss_points_Q4[i];
				double eta = d_gauss_points_Q4[j];
				double weight = d_gauss_weights_Q4[i] * d_gauss_weights_Q4[j];

				double dNdxi[4], dNdeta[4];
				shapeFunctionsQ4(xi, eta, dNdxi, dNdeta);

				double J[4] = { 0.0 };
				for (int k = 0; k < 4; k++) {
					J[0] += dNdxi[k] * elem.x[k]; J[1] += dNdxi[k] * elem.y[k];
					J[2] += dNdeta[k] * elem.x[k]; J[3] += dNdeta[k] * elem.y[k];
				}
				double detJ = J[0] * J[3] - J[1] * J[2];
				double invJ[4] = { J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ };

				double B[3 * 8] = { 0.0 };
				for (int k = 0; k < 4; k++) {
					double dNdx = invJ[0] * dNdxi[k] + invJ[1] * dNdeta[k];
					double dNdy = invJ[2] * dNdxi[k] + invJ[3] * dNdeta[k];
					B[0 * 8 + k * 2] = dNdx; B[1 * 8 + k * 2 + 1] = dNdy;
					B[2 * 8 + k * 2] = dNdy; B[2 * 8 + k * 2 + 1] = dNdx;
				}

				double int_weight = weight * detJ * d_THICKNESS;

				for (int r = 0; r < 8; r++) {
					for (int c = 0; c < 8; c++) {
						double sum = 0.0;
						for (int k = 0; k < 3; k++) {
							double d_val = 0.0;
							for (int l = 0; l < 3; l++) d_val += D[k * 3 + l] * B[l * 8 + c];
							sum += B[k * 8 + r] * d_val;
						}
						ke[r * 8 + c] += sum * int_weight;
					}
				}
			}
		}
	}
	else if (elem.type == ELEM_T3) {
		double dNdxi[3], dNdeta[3];
		shapeFunctionsT3(dNdxi, dNdeta);

		double J[4] = { 0.0 };
		for (int k = 0; k < 3; k++) {
			J[0] += dNdxi[k] * elem.x[k]; J[1] += dNdxi[k] * elem.y[k];
			J[2] += dNdeta[k] * elem.x[k]; J[3] += dNdeta[k] * elem.y[k];
		}
		double detJ = J[0] * J[3] - J[1] * J[2];
		double invJ[4] = { J[3] / detJ, -J[1] / detJ, -J[2] / detJ, J[0] / detJ };

		double int_weight = 0.5 * detJ * d_THICKNESS;

		double B[3 * 6] = { 0.0 };
		for (int k = 0; k < 3; k++) {
			double dNdx = invJ[0] * dNdxi[k] + invJ[1] * dNdeta[k];
			double dNdy = invJ[2] * dNdxi[k] + invJ[3] * dNdeta[k];
			B[0 * 6 + k * 2] = dNdx; B[1 * 6 + k * 2 + 1] = dNdy;
			B[2 * 6 + k * 2] = dNdy; B[2 * 6 + k * 2 + 1] = dNdx;
		}

		for (int r = 0; r < 6; r++) {
			for (int c = 0; c < 6; c++) {
				double sum = 0.0;
				for (int k = 0; k < 3; k++) {
					double d_val = 0.0;
					for (int l = 0; l < 3; l++) d_val += D[k * 3 + l] * B[l * 6 + c];
					sum += B[k * 6 + r] * d_val;
				}
				ke[r * MAX_ELEM_DOF + c] += sum * int_weight;
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

// ==================== A-LLS 核心 ====================
__device__ ALLSNode* d_alls_pool = nullptr;
__device__ int* d_alls_pool_index = nullptr;

__device__ ALLSNode* allocateALLSNode() {
	int index = atomicAdd(d_alls_pool_index, 1);
	return &d_alls_pool[index];
}

__global__ void assembleGlobalStiffnessKernelALLS(
	int num_elements,
	const Element* elements,
	ALLSNode** alls_row_heads,
	int* global_atomic_counter) {

	int eid = blockIdx.x * blockDim.x + threadIdx.x;
	if (eid >= num_elements) return;

	Element elem = elements[eid];
	double ke[MAX_ELEM_DOF * MAX_ELEM_DOF];
	computeElementStiffness(elem, ke); 

	int num_nodes = elem.num_nodes;
	int elem_dof = num_nodes * DOF_PER_NODE;
	int dof_indices[MAX_ELEM_DOF];
	for (int i = 0; i < num_nodes; i++) {
		dof_indices[i * 2] = elem.nodes[i] * 2;
		dof_indices[i * 2 + 1] = elem.nodes[i] * 2 + 1;
	}

	for (int i = 0; i < elem_dof; i++) {
		int row_global = dof_indices[i];
		if (row_global < 0) continue;
		for (int j = 0; j < elem_dof; j++) {
			int col_global = dof_indices[j];
			if (col_global < 0) continue;
			double value = ke[i * MAX_ELEM_DOF + j];

			ALLSNode* head = alls_row_heads[row_global];
			ALLSNode* current = head;
			ALLSNode* prev = nullptr;
			bool found = false;

			while (current != nullptr) {
				if (current->col == col_global) {
					atomicAdd(&(current->value), value);
					found = true;
					break;
				}
				prev = current;
				current = current->next;
			}

			if (!found) {
				ALLSNode* newNode = allocateALLSNode();
				newNode->col = col_global;
				newNode->value = value;
				newNode->next = nullptr;

				if (prev == nullptr) {
					ALLSNode* oldHead = (ALLSNode*)atomicExch((unsigned long long int*)&alls_row_heads[row_global],
						(unsigned long long int)newNode);
					newNode->next = oldHead;
				}
				else {
					ALLSNode* oldNext = (ALLSNode*)atomicExch((unsigned long long int*)&prev->next,
						(unsigned long long int)newNode);
					newNode->next = oldNext;
				}
				atomicAdd(global_atomic_counter, 1);
			}
		}
	}
}

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

	// Note: 避免在这重复清零 csr_values，因为有可能是并发累加。
	// 在 Host 端用 cudaMemset 清零是最安全的。
	// for (int i = row_start; i <= row_end; i++) csr_values[i] = 0.0;

	ALLSNode* current = alls_row_heads[row];
	while (current != nullptr) {
		int pos = binary_search(csr_col_ind, row_start, row_end, current->col);
		if (pos >= 0) {
			atomicAdd(&csr_values[pos], current->value);
		}
		current = current->next;
	}
}

__global__ void assembleGlobalStiffnessKernelColored(
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

	int elements_in_color = color_sizes[color_id];
	int start_offset = color_offsets[color_id];

	for (int idx = threadIdx.x; idx < elements_in_color; idx += blockDim.x) {
		int eid = start_offset + idx;
		if (eid >= num_elements) continue;

		Element elem = elements[eid];
		double ke[MAX_ELEM_DOF * MAX_ELEM_DOF];
		computeElementStiffness(elem, ke); // Load Imbalance source in Mixed Mesh

		int num_nodes = elem.num_nodes;
		int elem_dof = num_nodes * DOF_PER_NODE;
		int dof_indices[MAX_ELEM_DOF];

		for (int i = 0; i < num_nodes; i++) {
			dof_indices[i * 2] = elem.nodes[i] * 2;
			dof_indices[i * 2 + 1] = elem.nodes[i] * 2 + 1;
		}

		for (int i = 0; i < elem_dof; i++) {
			int row_global = dof_indices[i];
			int row_start = csr_row_ptr[row_global];
			int row_end = csr_row_ptr[row_global + 1] - 1;

			for (int j = 0; j < elem_dof; j++) {
				int col_global = dof_indices[j];
				int pos = binary_search(csr_col_ind, row_start, row_end, col_global);
				if (pos >= 0) {
					csr_values[pos] += ke[i * MAX_ELEM_DOF + j];
				}
			}
		}
	}
}

// ==================== 网格生成与工具 ====================
class MeshGenerator {
public:
	static void generateStructuredQ4(int nx, int ny, double lx, double ly,
		std::vector<Node>& nodes, std::vector<Element>& elements) {
		nodes.resize((nx + 1) * (ny + 1));
		for (int j = 0; j <= ny; j++) {
			for (int i = 0; i <= nx; i++) {
				int nid = j * (nx + 1) + i;
				nodes[nid].x = i * lx / nx; nodes[nid].y = j * ly / ny;
			}
		}
		elements.resize(nx * ny);
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				int eid = j * nx + i;
				Element& e = elements[eid];
				e.type = ELEM_Q4; e.num_nodes = 4; e.color = -1;
				e.nodes[0] = j * (nx + 1) + i; e.nodes[1] = j * (nx + 1) + i + 1;
				e.nodes[2] = (j + 1) * (nx + 1) + i + 1; e.nodes[3] = (j + 1) * (nx + 1) + i;
				for (int k = 0; k < 4; k++) { e.x[k] = nodes[e.nodes[k]].x; e.y[k] = nodes[e.nodes[k]].y; }
			}
		}
	}

	static void generateStructuredT3(int nx, int ny, double lx, double ly,
		std::vector<Node>& nodes, std::vector<Element>& elements) {
		nodes.resize((nx + 1) * (ny + 1));
		for (int j = 0; j <= ny; j++) {
			for (int i = 0; i <= nx; i++) {
				int nid = j * (nx + 1) + i;
				nodes[nid].x = i * lx / nx; nodes[nid].y = j * ly / ny;
			}
		}
		elements.resize(nx * ny * 2);
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				int base_eid = (j * nx + i) * 2;
				int n0 = j * (nx + 1) + i; int n1 = j * (nx + 1) + i + 1;
				int n2 = (j + 1) * (nx + 1) + i + 1; int n3 = (j + 1) * (nx + 1) + i;

				Element& e1 = elements[base_eid];
				e1.type = ELEM_T3; e1.num_nodes = 3; e1.color = -1;
				e1.nodes[0] = n0; e1.nodes[1] = n1; e1.nodes[2] = n2; e1.nodes[3] = -1;
				Element& e2 = elements[base_eid + 1];
				e2.type = ELEM_T3; e2.num_nodes = 3; e2.color = -1;
				e2.nodes[0] = n0; e2.nodes[1] = n2; e2.nodes[2] = n3; e2.nodes[3] = -1;
				for (int k = 0; k < 3; k++) {
					e1.x[k] = nodes[e1.nodes[k]].x; e1.y[k] = nodes[e1.nodes[k]].y;
					e2.x[k] = nodes[e2.nodes[k]].x; e2.y[k] = nodes[e2.nodes[k]].y;
				}
			}
		}
	}

	static void generateMixedMesh(int nx, int ny, double lx, double ly,
		std::vector<Node>& nodes, std::vector<Element>& elements, double q4_ratio = 0.5) {
		nodes.resize((nx + 1) * (ny + 1));
		for (int j = 0; j <= ny; j++) {
			for (int i = 0; i <= nx; i++) {
				int nid = j * (nx + 1) + i;
				nodes[nid].x = i * lx / nx; nodes[nid].y = j * ly / ny;
			}
		}
		elements.reserve(nx * ny * 2);
		std::mt19937 rng(42);
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				int n0 = j * (nx + 1) + i; int n1 = j * (nx + 1) + i + 1;
				int n2 = (j + 1) * (nx + 1) + i + 1; int n3 = (j + 1) * (nx + 1) + i;
				if (dist(rng) < q4_ratio) {
					Element e;
					e.type = ELEM_Q4; e.num_nodes = 4; e.color = -1;
					e.nodes[0] = n0; e.nodes[1] = n1; e.nodes[2] = n2; e.nodes[3] = n3;
					for (int k = 0; k < 4; k++) { e.x[k] = nodes[e.nodes[k]].x; e.y[k] = nodes[e.nodes[k]].y; }
					elements.push_back(e);
				}
				else {
					Element e1, e2;
					e1.type = ELEM_T3; e1.num_nodes = 3; e1.color = -1;
					e1.nodes[0] = n0; e1.nodes[1] = n1; e1.nodes[2] = n2; e1.nodes[3] = -1;
					e2.type = ELEM_T3; e2.num_nodes = 3; e2.color = -1;
					e2.nodes[0] = n0; e2.nodes[1] = n2; e2.nodes[2] = n3; e2.nodes[3] = -1;
					for (int k = 0; k < 3; k++) {
						e1.x[k] = nodes[e1.nodes[k]].x; e1.y[k] = nodes[e1.nodes[k]].y;
						e2.x[k] = nodes[e2.nodes[k]].x; e2.y[k] = nodes[e2.nodes[k]].y;
					}
					elements.push_back(e1); elements.push_back(e2);
				}
			}
		}
	}
};

void buildCSRStructure(int num_nodes, const std::vector<Element>& elements,
	std::vector<int>& row_ptr, std::vector<int>& col_ind) {
	int total_dofs = num_nodes * DOF_PER_NODE;
	std::vector<std::set<int>> column_indices(total_dofs);
	for (const auto& elem : elements) {
		int dof_indices[MAX_ELEM_DOF];
		for (int i = 0; i < elem.num_nodes; i++) {
			dof_indices[i * 2] = elem.nodes[i] * 2;
			dof_indices[i * 2 + 1] = elem.nodes[i] * 2 + 1;
		}
		int elem_dof = elem.num_nodes * DOF_PER_NODE;
		for (int i = 0; i < elem_dof; i++) {
			for (int j = 0; j < elem_dof; j++) {
				column_indices[dof_indices[i]].insert(dof_indices[j]);
			}
		}
	}
	row_ptr.resize(total_dofs + 1);
	row_ptr[0] = 0;
	for (int i = 0; i < total_dofs; i++) row_ptr[i + 1] = row_ptr[i] + column_indices[i].size();
	int nnz = row_ptr[total_dofs];
	col_ind.resize(nnz);
	for (int i = 0; i < total_dofs; i++)
		std::copy(column_indices[i].begin(), column_indices[i].end(), col_ind.begin() + row_ptr[i]);
}

int colorElements(std::vector<Element>& elements, std::vector<std::vector<int>>& color_groups) {
	int num_elements = elements.size();
	std::vector<std::set<int>> adjacents(num_elements);
	std::map<int, std::vector<int>> node_to_elements;

	for (int eid = 0; eid < num_elements; eid++) {
		for (int i = 0; i < elements[eid].num_nodes; i++) {
			node_to_elements[elements[eid].nodes[i]].push_back(eid);
		}
	}
	for (auto const& pair : node_to_elements) {
		const std::vector<int>& eids = pair.second;
		for (size_t i = 0; i < eids.size(); i++) {
			for (size_t j = i + 1; j < eids.size(); j++) {
				adjacents[eids[i]].insert(eids[j]);
				adjacents[eids[j]].insert(eids[i]);
			}
		}
	}
	std::vector<int> colors(num_elements, -1);
	int num_colors = 0;
	for (int eid = 0; eid < num_elements; eid++) {
		std::vector<bool> used(num_colors + 1, false);
		for (int adj : adjacents[eid]) {
			if (colors[adj] != -1) {
				if (colors[adj] < used.size()) used[colors[adj]] = true;
			}
		}
		int c = 0;
		while (c < used.size() && used[c]) c++;
		colors[eid] = c;
		if (c >= num_colors) num_colors = c + 1;
	}
	color_groups.resize(num_colors);
	for (int eid = 0; eid < num_elements; eid++) {
		elements[eid].color = colors[eid];
		color_groups[colors[eid]].push_back(eid);
	}
	return num_colors;
}

LoadStats analyzeLoadImbalance(const std::vector<Element>& elements) {
	LoadStats stats;
	stats.count_q4 = 0;
	stats.count_t3 = 0;

	for (const auto& e : elements) {
		if (e.type == ELEM_Q4) stats.count_q4++;
		else if (e.type == ELEM_T3) stats.count_t3++;
	}

	int total = elements.size();
	stats.percent_q4 = (double)stats.count_q4 / total * 100.0;
	stats.percent_t3 = (double)stats.count_t3 / total * 100.0;

	stats.avg_compute_intensity = (stats.count_q4 * GAUSS_POINTS_Q4 + stats.count_t3 * GAUSS_POINTS_T3) / (double)total;

	if (stats.count_q4 == total) stats.load_description = "Uniform Heavy (All Q4)";
	else if (stats.count_t3 == total) stats.load_description = "Uniform Light (All T3)";
	else stats.load_description = "Mixed/Imbalanced";

	return stats;
}

PerformanceMetrics runComparison(
	std::vector<Element>& elements,
	int num_nodes,
	const std::vector<int>& csr_row_ptr,
	const std::vector<int>& csr_col_ind) {

	PerformanceMetrics metrics;
	int num_elements = elements.size();
	int total_dofs = num_nodes * DOF_PER_NODE;
	int nnz = csr_row_ptr[total_dofs];

	// 1. 染色
	auto t1 = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<int>> color_groups;
	int num_colors = colorElements(elements, color_groups);
	auto t2 = std::chrono::high_resolution_clock::now();
	metrics.coloring_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;

	std::vector<int> sorted_indices;
	std::vector<int> color_offsets(num_colors);
	std::vector<int> color_sizes(num_colors);
	int offset = 0;
	for (int c = 0; c < num_colors; c++) {
		color_offsets[c] = offset;
		color_sizes[c] = color_groups[c].size();
		sorted_indices.insert(sorted_indices.end(), color_groups[c].begin(), color_groups[c].end());
		offset += color_sizes[c];
	}
	std::vector<Element> sorted_elements(num_elements);
	for (int i = 0; i < num_elements; i++) sorted_elements[i] = elements[sorted_indices[i]];

	Element* d_elements;
	int *d_csr_row_ptr, *d_csr_col_ind;
	double *d_vals_alls, *d_vals_color;
	int *d_c_offsets, *d_c_sizes;

	CHECK_CUDA_ERROR(cudaMalloc(&d_elements, num_elements * sizeof(Element)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_row_ptr, (total_dofs + 1) * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_vals_alls, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_vals_color, nnz * sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_c_offsets, num_colors * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_c_sizes, num_colors * sizeof(int)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_elements, sorted_elements.data(), num_elements * sizeof(Element), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (total_dofs + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_c_offsets, color_offsets.data(), num_colors * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_c_sizes, color_sizes.data(), num_colors * sizeof(int), cudaMemcpyHostToDevice));

	ALLSNode** d_alls_heads;
	ALLSNode* d_alls_pool;
	int* d_alls_idx;
	int* d_atomic_counter;

	CHECK_CUDA_ERROR(cudaMalloc(&d_alls_heads, total_dofs * sizeof(ALLSNode*)));
	size_t pool_size = nnz * 2;
	CHECK_CUDA_ERROR(cudaMalloc(&d_alls_pool, pool_size * sizeof(ALLSNode)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_alls_idx, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_atomic_counter, sizeof(int)));

	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool, &d_alls_pool, sizeof(ALLSNode*)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::d_alls_pool_index, &d_alls_idx, sizeof(int*)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	// A-LLS
	int grid_dim = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int grid_conv = (total_dofs + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// 记录内存使用
	size_t mem_free_start, mem_total, mem_free_end;
	cudaMemGetInfo(&mem_free_start, &mem_total);

	// -- A-LLS 预热迭代 --
	for (int i = 0; i < WARMUP_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_alls_heads, 0, total_dofs * sizeof(ALLSNode*)));
		CHECK_CUDA_ERROR(cudaMemset(d_alls_idx, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_atomic_counter, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_vals_alls, 0, nnz * sizeof(double)));

		assembleGlobalStiffnessKernelALLS <<<grid_dim, BLOCK_SIZE >>> (num_elements, d_elements, d_alls_heads, d_atomic_counter);
		convertALLStoCSRKernel <<<grid_conv, BLOCK_SIZE >>> (total_dofs, d_alls_heads, d_vals_alls, d_csr_row_ptr, d_csr_col_ind);
	}
	cudaDeviceSynchronize();

	// -- A-LLS 测量迭代 --
	float total_alls_ms = 0.0f;
	for (int i = 0; i < MEASURE_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_alls_heads, 0, total_dofs * sizeof(ALLSNode*)));
		CHECK_CUDA_ERROR(cudaMemset(d_alls_idx, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_atomic_counter, 0, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_vals_alls, 0, nnz * sizeof(double)));

		cudaEventRecord(start);
		assembleGlobalStiffnessKernelALLS <<<grid_dim, BLOCK_SIZE >>> (num_elements, d_elements, d_alls_heads, d_atomic_counter);
		convertALLStoCSRKernel <<<grid_conv, BLOCK_SIZE >>> (total_dofs, d_alls_heads, d_vals_alls, d_csr_row_ptr, d_csr_col_ind);
		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		float iter_ms;
		cudaEventElapsedTime(&iter_ms, start, stop);
		total_alls_ms += iter_ms;
	}
	metrics.alls_time_ms = total_alls_ms / MEASURE_ITERS;
	
	cudaMemGetInfo(&mem_free_end, &mem_total);
	metrics.alls_mem_usage = mem_free_start - mem_free_end;
	CHECK_CUDA_ERROR(cudaMemcpy(&metrics.atomic_ops_alls, d_atomic_counter, sizeof(int), cudaMemcpyDeviceToHost));


	// Coloring
	cudaMemGetInfo(&mem_free_start, &mem_total);
	
	int max_color_size = 0;
	for (int s : color_sizes) max_color_size = std::max(max_color_size, s);
	int threads_per_block = std::min(256, max_color_size);
	if (threads_per_block < 32) threads_per_block = 32;

	// -- Coloring 预热迭代 --
	for (int i = 0; i < WARMUP_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_vals_color, 0, nnz * sizeof(double)));
		assembleGlobalStiffnessKernelColored <<<num_colors, threads_per_block >>> (
			num_elements, d_elements, d_vals_color, d_csr_row_ptr, d_csr_col_ind,
			d_c_offsets, d_c_sizes, num_colors, total_dofs, nnz
		);
	}
	cudaDeviceSynchronize();

	// -- Coloring 测量迭代 --
	float total_col_ms = 0.0f;
	for (int i = 0; i < MEASURE_ITERS; i++) {
		CHECK_CUDA_ERROR(cudaMemset(d_vals_color, 0, nnz * sizeof(double)));
		cudaEventRecord(start);
		assembleGlobalStiffnessKernelColored <<<num_colors, threads_per_block >>> (
			num_elements, d_elements, d_vals_color, d_csr_row_ptr, d_csr_col_ind,
			d_c_offsets, d_c_sizes, num_colors, total_dofs, nnz
		);
		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		float iter_ms;
		cudaEventElapsedTime(&iter_ms, start, stop);
		total_col_ms += iter_ms;
	}
	metrics.colored_time_ms = total_col_ms / MEASURE_ITERS;

	cudaMemGetInfo(&mem_free_end, &mem_total);
	metrics.colored_mem_usage = mem_free_start - mem_free_end;

	metrics.speedup = metrics.colored_time_ms / metrics.alls_time_ms;

	cudaFree(d_elements); cudaFree(d_csr_row_ptr); cudaFree(d_csr_col_ind);
	cudaFree(d_vals_alls); cudaFree(d_vals_color);
	cudaFree(d_c_offsets); cudaFree(d_c_sizes);
	cudaFree(d_alls_heads); cudaFree(d_alls_pool);
	cudaFree(d_alls_idx); cudaFree(d_atomic_counter);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	return metrics;
}

// ==================== 主函数 ====================
int main() {
	std::cout << "Starting Comprehensive FEM Assembly Benchmark with Load Imbalance Analysis..." << std::endl;
	std::cout << "Using Warm-up iterations: " << WARMUP_ITERS << ", Measurement iterations: " << MEASURE_ITERS << "\n" << std::endl;

	int sizes[] = { 50, 100, 200, 400, 800 };
	std::vector<TestConfig> results;

	for (int size : sizes) {
		int nx = size, ny = size;

		// 1. Structured Q4
		{
			TestConfig conf;
			conf.name = "Q4_" + std::to_string(nx) + "x" + std::to_string(ny);
			std::vector<Node> nodes; std::vector<Element> elems;
			MeshGenerator::generateStructuredQ4(nx, ny, 1.0, 1.0, nodes, elems);
			conf.num_elements = elems.size();
			conf.num_nodes = nodes.size();
			conf.load_stats = analyzeLoadImbalance(elems);

			std::vector<int> row_ptr, col_ind;
			buildCSRStructure(nodes.size(), elems, row_ptr, col_ind);
			conf.metrics = runComparison(elems, nodes.size(), row_ptr, col_ind);
			results.push_back(conf);
		}

		// 2. Structured T3
		{
			TestConfig conf;
			conf.name = "T3_" + std::to_string(nx) + "x" + std::to_string(ny);
			std::vector<Node> nodes; std::vector<Element> elems;
			MeshGenerator::generateStructuredT3(nx, ny, 1.0, 1.0, nodes, elems);
			conf.num_elements = elems.size();
			conf.num_nodes = nodes.size();
			conf.load_stats = analyzeLoadImbalance(elems);

			std::vector<int> row_ptr, col_ind;
			buildCSRStructure(nodes.size(), elems, row_ptr, col_ind);
			conf.metrics = runComparison(elems, nodes.size(), row_ptr, col_ind);
			results.push_back(conf);
		}

		// 3. Mixed Mesh (Load Imbalance)
		{
			TestConfig conf;
			conf.name = "Mixed_" + std::to_string(nx) + "x" + std::to_string(ny);
			std::vector<Node> nodes; std::vector<Element> elems;
			MeshGenerator::generateMixedMesh(nx, ny, 1.0, 1.0, nodes, elems, 0.5);
			conf.num_elements = elems.size();
			conf.num_nodes = nodes.size();
			conf.load_stats = analyzeLoadImbalance(elems);

			std::vector<int> row_ptr, col_ind;
			buildCSRStructure(nodes.size(), elems, row_ptr, col_ind);
			conf.metrics = runComparison(elems, nodes.size(), row_ptr, col_ind);
			results.push_back(conf);
		}
	}

	// --- 增强的输出，针对 Load Imbalance 分析 ---
	std::cout << "\n=====================================================================" << std::endl;
	std::cout << "               LOAD IMBALANCE & PERFORMANCE ANALYSIS                 " << std::endl;
	std::cout << "=====================================================================" << std::endl;
	std::cout << std::left << std::setw(15) << "TestName"
		<< std::setw(12) << "Elements"
		<< std::setw(10) << "Q4(%)"
		<< std::setw(10) << "T3(%)"
		<< std::setw(20) << "Load Description"
		<< std::setw(10) << "AvgLoad"
		<< std::setw(12) << "A-LLS(ms)"
		<< std::setw(12) << "Color(ms)"
		<< std::setw(10) << "Speedup"
		<< std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;

	for (const auto& res : results) {
		std::cout << std::left << std::setw(15) << res.name
			<< std::setw(12) << res.num_elements
			<< std::setw(10) << std::fixed << std::setprecision(1) << res.load_stats.percent_q4
			<< std::setw(10) << std::fixed << std::setprecision(1) << res.load_stats.percent_t3
			<< std::setw(20) << res.load_stats.load_description
			<< std::setw(10) << std::fixed << std::setprecision(2) << res.load_stats.avg_compute_intensity
			<< std::setw(12) << std::setprecision(3) << res.metrics.alls_time_ms
			<< std::setw(12) << std::setprecision(3) << res.metrics.colored_time_ms
			<< std::setw(10) << std::setprecision(2) << res.metrics.speedup
			<< std::endl;
	}

	std::cout << "\nAnalysis Notes:" << std::endl;
	std::cout << "1. 'AvgLoad': Average computational intensity per element (based on integration points)." << std::endl;
	std::cout << "   - Q4 (4 points) is significantly heavier than T3 (1 point)." << std::endl;
	std::cout << "2. 'Mixed' scenarios represent high load imbalance within thread warps." << std::endl;
	std::cout << "   - In Coloring method, T3 threads must wait for Q4 threads in the same color group." << std::endl;
	std::cout << "   - A-LLS's speedup in Mixed scenarios demonstrates robustness against this imbalance." << std::endl;

	// CSV 输出
	std::ofstream csv("benchmark_results_v2.csv");
	csv << "Scenario,NumElements,Percent_Q4,Percent_T3,Avg_Compute_Load,Time_ALLS_ms,Time_Color_ms,Speedup\n";
	for (const auto& res : results) {
		csv << res.name << ","
			<< res.num_elements << ","
			<< res.load_stats.percent_q4 << ","
			<< res.load_stats.percent_t3 << ","
			<< res.load_stats.avg_compute_intensity << ","
			<< res.metrics.alls_time_ms << ","
			<< res.metrics.colored_time_ms << ","
			<< res.metrics.speedup << "\n";
	}
	csv.close();
	std::cout << "Saved detailed results to benchmark_results_v2.csv" << std::endl;

	return 0;
}
