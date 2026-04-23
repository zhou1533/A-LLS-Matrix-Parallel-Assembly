This is a C++ program for GPU parallel assembly of the global stiffness matrix in finite element analysis, using the Atomic Operation Linked List Structure (A-LLS) method and written in CUDA. The approach introduces row-wise linked lists as an intermediate data structure, where each thread uses atomic operations to safely insert or accumulate element contributions into the linked list of the corresponding row. Subsequently, a specially designed post-processing kernel efficiently consolidates the linked list data into the final matrix in CSR (Compressed Sparse Row) format. Within this framework, all elements can be processed in fully parallel without the need for coloring preprocessing, fundamentally avoiding the load imbalance and synchronization overhead issues associated with traditional coloring methods.

License
This project is licensed under the MIT License.

Numerical Examples	in the article：
 
Numerical Example 1 ：Quadrilateral Mesh ALLS_Quad2D_Assembly_Accuracy_Analysis.cu and ALLS_Quad2D_Assembly_Performance_Analysis.cu	（2-D）
Numerical Example 2：Unstructured Triangular Mesh	ALLS_T3_Unstructed_Elements_Assemble_Performanc.cu	（2-D）
Numerical 3：Hybrid Mesh	ALLS_Q4_T3_Mixed_Elements_Assembly_Performance.cu	（2-D）
Numerical Example 4：Hexahedral Mesh ALLS_Hex20_3D_Assembly_Performance.cu	（3-D）

