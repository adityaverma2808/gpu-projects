/*
  CS 6023 Assignment 3. 
  Do not make any changes to the boiler plate code or the other files in the folder.
  Use cudaFree to deallocate any memory not in usage.
  Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <vector>

using namespace std;

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
  /* Function for parsing input file*/

  FILE *inputFile = NULL;
  // Read the file for input. 
  if ((inputFile = fopen (fileName, "r")) == NULL) {
    printf ("Failed at opening the file %s\n", fileName) ;
    return ;
  }

  // Input the header information.
  int numMeshes ;
  fscanf (inputFile, "%d", &numMeshes) ;
  fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
  

  // Input all meshes and store them inside a vector.
  int meshX, meshY ;
  int globalPositionX, globalPositionY; // top left corner of the matrix.
  int opacity ;
  int* currMesh ;
  for (int i=0; i<numMeshes; i++) {
    fscanf (inputFile, "%d %d", &meshX, &meshY) ;
    fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
    fscanf (inputFile, "%d", &opacity) ;
    currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
    for (int j=0; j<meshX; j++) {
      for (int k=0; k<meshY; k++) {
        fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
      }
    }
    //Create a Scene out of the mesh.
    SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
    scenes.push_back (scene) ;
  }

  // Input all relations and store them in edges.
  int relations;
  fscanf (inputFile, "%d", &relations) ;
  int u, v ; 
  for (int i=0; i<relations; i++) {
    fscanf (inputFile, "%d %d", &u, &v) ;
    edges.push_back ({u,v}) ;
  }

  // Input all translations.
  int numTranslations ;
  fscanf (inputFile, "%d", &numTranslations) ;
  std::vector<int> command (3, 0) ;
  for (int i=0; i<numTranslations; i++) {
    fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
    translations.push_back (command) ;
  }
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
  /* Function for writing the final png into a file.*/
  FILE *outputFile = NULL; 
  if ((outputFile = fopen (outputFileName, "w")) == NULL) {
    printf ("Failed while opening output file\n") ;
  }
  
  for (int i=0; i<frameSizeX; i++) {
    for (int j=0; j<frameSizeY; j++) {
      fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
    }
    fprintf (outputFile, "\n") ;
  }
}
__global__ void fillScene(int len_x, int len_y, int frame_x, int frame_y, int opa, int meshIdx, int* mesh, int* globalPositionX, int* globalPositionY, int *GlobalScene, int* GlobalSceneopa){

    int grid_no = blockIdx.x * blockDim.x;
    int row = (grid_no + threadIdx.x) / len_y;
    int col = (grid_no + threadIdx.x) % len_y;

    if(row >= 0 && col >= 0){
      if(row < len_x && col < len_y){
        int globalX = row;
        globalX+= globalPositionX[meshIdx];
        int globalY = col;
        globalY+= globalPositionY[meshIdx];
        if (globalX < frame_x && globalY < frame_y)
        {
            if(globalX >= 0 && globalY >= 0 ){
              int key = globalX * frame_y + globalY;
              int mesh_key = row * len_y + col;
              if (GlobalSceneopa[key] < opa)
              {
                  GlobalScene[key] = mesh[mesh_key];
                  GlobalSceneopa[key] = opa;
              }
            }
        }
      }
    }

}

__device__ void solve(int m,int dir,int amount,int* device_x,int* device_y,int* device_csr,int *device_off,int V)
{

  if(dir==3){
    atomicAdd(&device_y[m],amount);
  }else if(dir==2) {
    atomicAdd(&device_y[m],-amount);
  }else if(dir==1) {
    atomicAdd(&device_x[m],amount);
  }else{
    atomicAdd(&device_x[m],-amount);
  }

  int start,end;
  start = device_off[m];
  end = device_off[m+1];
  int child = start;
  for(;child<end;)
  {
    solve(device_csr[child],dir,amount,device_x,device_y,device_csr,device_off,V);
    child++;
  }  
  
}

__global__ void apply_trans(int translation,int* device_tr,int* device_x,int* device_y,int* device_csr,int *device_off,int V)
{
  if((blockIdx.x*blockDim.x+threadIdx.x)<translation)
  {
    int node_key = 3*(blockIdx.x*blockDim.x+threadIdx.x);
    int desc_key = 3*(blockIdx.x*blockDim.x+threadIdx.x)+1;
    int amount_key = 3*(blockIdx.x*blockDim.x+threadIdx.x)+2;
    solve(device_tr[node_key],device_tr[desc_key],device_tr[amount_key],device_x,device_y,device_csr,device_off,V);
  }
}



int main (int argc, char **argv) {
  
  // Read the scenes into memory from File.
  const char *inputFileName = argv[1] ;
  int* hFinalPng ; 
  const char *outputFileName = argv[2] ;
    FILE *outputFile = NULL; 
  if ((outputFile = fopen (outputFileName, "w")) == NULL) {
    printf ("Failed while opening output file\n") ;
  }

  int frameSizeX, frameSizeY ;
  std::vector<SceneNode*> scenes ;
  std::vector<std::vector<int> > edges ;
  std::vector<std::vector<int> > translations ;
  readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
  hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
  
  // Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

  // Basic information.
  int V = scenes.size () ;
  int E = edges.size () ;
  int numTranslations = translations.size () ;

  // Convert the scene graph into a csr.
  scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
  int *hOffset = scene->get_h_offset () ;  
  int *hCsr = scene->get_h_csr () ;
  int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
  int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
  int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
  int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
  int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
  int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

  auto start = std::chrono::high_resolution_clock::now () ;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Code begins here.
  // Do not change anything above this comment.
    int total_blocks = ceil(numTranslations/1024.0);
    int total_threads = 1024;
    int key = 0;
    int vec_len = sizeof(int)*V;
    int offset_len = sizeof(int)*(V+1);
    
    int* host_tr = (int*)malloc(sizeof(int)*numTranslations*3);
    int i=0;
    for(;i<numTranslations;i+=1,key+=3){
      host_tr[key] = translations[i][0];
      int query_key = key + 1; 
      host_tr[query_key] = translations[i][1];
      int amount_key = key + 2;
      host_tr[amount_key] = translations[i][2];
    } 
    int* device_tr;
    int translationSize = sizeof(int)*numTranslations*3;
    cudaMalloc(&device_tr,translationSize);
    cudaMemcpy(device_tr,host_tr,translationSize,cudaMemcpyHostToDevice);

    int* device_x,*device_y;

    cudaMalloc(&device_x,vec_len);
    cudaMemcpy(device_x,hGlobalCoordinatesX,vec_len,cudaMemcpyHostToDevice);

    cudaMalloc(&device_y,vec_len);
    cudaMemcpy(device_y,hGlobalCoordinatesY,vec_len,cudaMemcpyHostToDevice);


    int *device_off;
    cudaMalloc(&device_off,offset_len);
    cudaMemcpy(device_off,hOffset,offset_len,cudaMemcpyHostToDevice);

    int* device_csr;
    cudaMalloc(&device_csr,offset_len);
    cudaMemcpy(device_csr,hCsr,offset_len,cudaMemcpyHostToDevice);

    apply_trans<<<total_blocks,total_threads>>>(numTranslations,device_tr,device_x,device_y,device_csr,device_off,V);

    cudaDeviceSynchronize();
    cudaMemcpy(hGlobalCoordinatesX,device_x,vec_len,cudaMemcpyDeviceToHost);
    cudaMemcpy(hGlobalCoordinatesY,device_y,vec_len,cudaMemcpyDeviceToHost);

    int *dGlobalScene;
    int *dGlobalSceneOpacity;
    int gloabl_scene_size = sizeof(int) * frameSizeX * frameSizeY;

    cudaMalloc(&dGlobalScene, gloabl_scene_size);
    cudaMalloc(&dGlobalSceneOpacity, gloabl_scene_size);
    cudaMemset(&dGlobalScene, 0, gloabl_scene_size);

    cudaMemset(&dGlobalSceneOpacity, -1, gloabl_scene_size);



    int ii = 0;
    for (; ii < V; ii+=1) {
        int meshY = hFrameSizeY[ii];
        int orginal_mesh_x = hFrameSizeX[ii];
        unsigned int opacity = hOpacity[ii];
        int* dMesh;
        int mesh_size = sizeof(int) * orginal_mesh_x * meshY;
        cudaMalloc(&dMesh, mesh_size);
        cudaMemcpy(dMesh, hMesh[ii], mesh_size, cudaMemcpyHostToDevice);
        int threads = 1024;
        fillScene<<<ceil(float(orginal_mesh_x*meshY)/ 1024.0), threads>>>(orginal_mesh_x, meshY, frameSizeX, frameSizeY, opacity, ii, dMesh, device_x,device_y, dGlobalScene, dGlobalSceneOpacity);

    }
    cudaMemcpy(hFinalPng, dGlobalScene, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);

  // Do not change anything below this comment.
  // Code ends here.

  auto end  = std::chrono::high_resolution_clock::now () ;

  std::chrono::duration<double, std::micro> timeTaken = end-start;

  printf ("execution time : %f\n", timeTaken) ;
  writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ; 

}
