// On resout l'equation de la chaleur (de Laplace) sur un domaine regulier.
#include <iostream>
#include <iterator>
#include <fstream>
#include "Array2D.hpp"
#include <mpi.h>
#include <time.h>
#include <string>

// Sauvegarde d'une matrice dans un fichier texte
void save(Array2D<double> &matrix, std::string name) {
  std::ofstream file(name.c_str());
  for (int iY=0; iY<matrix.sizeY(); ++iY) {
     copy(&matrix.data()[iY*matrix.sizeX()], &matrix.data()[iY*matrix.sizeX()]+matrix.sizeX(),
          std::ostream_iterator<double>(file, " "));
     file << "\n";
  }
}



int main(int argc , char** argv) {

  const int NX = 1000;
  const int NY = 1000;
  const int maxT = 40000;


  int myRank,nProc;
  MPI_Init(&argc, &argv);
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProc);

  int dimX = NX;
  int dimY = NY/nProc; // assuming perfect division



  Array2D<double> heat(dimX, dimY, 0); // La matrice de la chaleur
  Array2D<double> tmp(dimX, dimY, 0);  // Une matrice temporaire
  std::vector<double> recv_upper(dimX,0);
  std::vector<double> recv_lower(dimX,0);

  
  
  if(myRank==0){
  for (int iX=0; iX<dimX; iX++) {      // conditions aux bords:
      heat(iX,0) = 0;                 // 0 en haut
      tmp(iX,0) = 0;                  // 0 en haut
      
  }
  }



  if(myRank== nProc-1){
    for (int iX=0; iX<dimX; iX++) {      // conditions aux bords:
      heat(iX,dimY-1) = 1;            // 1 en bas
      tmp(iX,dimY-1) = 1;             // 1 en bas
  }
  }  

  for (int iY=0; iY<dimY; iY++) {
      heat(0,iY)      = 0.;           // 0 a gauche
      heat(dimX-1,iY) = 1.;           // 1 a droite
      tmp(0,iY)      = 0.;            // 0 a gauche
      tmp(dimX-1,iY) = 1.;            // 1 a droite
  }

  MPI_Request request;
  
  float  comp_time, io_time, total_time;

  clock_t timepoint = clock();

  comp_time =  (float)(clock() - timepoint) / CLOCKS_PER_SEC;
  for (int iT=0; iT<maxT; iT++) { 
    
    double* upper = heat.data();
    double* lower = heat.data() + (dimY-2)*dimX;

    if(myRank>0)
    {  
      MPI_Isend( upper, dimX , MPI_DOUBLE , myRank-1  , 1 , MPI_COMM_WORLD ,  &request);
      MPI_Irecv( recv_lower.data() ,dimX,  MPI_DOUBLE , myRank-1  , 0 , MPI_COMM_WORLD ,  &request);
    }
      if(myRank<nProc-1)
    {  
      MPI_Isend( lower, dimX , MPI_DOUBLE , myRank+1  , 0 , MPI_COMM_WORLD ,  &request);
      MPI_Irecv( recv_upper.data() ,dimX,  MPI_DOUBLE , myRank+1  , 1 , MPI_COMM_WORLD ,  &request);
    }
    
         // do computation for part that doesnt need comms (save time)
    for (int iY=1; iY<dimY-1; iY++) {  
      for (int iX=1; iX<dimX-1; iX++) {
        tmp(iX,iY) = 0.25*( heat(iX-1,iY) + heat(iX+1,iY)+
                            heat(iX,iY-1) + heat(iX,iY+1) );
      }
    }
  
    //check finish comms
    MPI_Barrier( MPI_COMM_WORLD);

    // finish calculation for comm halo upper
    int iY=0;
    if(myRank>0)
    for (int iX=1; iX<dimX-1; iX++)
      tmp(iX,iY) = 0.25*( heat(iX-1,iY) + heat(iX+1,iY) +  recv_upper[iX] + heat(iX,iY+1) );
    
    // finish calculation for comm halo lower
    iY=dimY-1;
    if(myRank<nProc-1)
    for (int iX=1; iX<dimX-1; iX++)
      tmp(iX,iY) = 0.25*( heat(iX-1,iY) + heat(iX+1,iY) +  heat(iX,iY-1) + recv_lower[iX] );

    heat.unsafeSwap(tmp);              
  }
  comp_time =  (float)(clock() - timepoint) / CLOCKS_PER_SEC;
  timepoint = clock();
  save(heat, "chaleur"+ std::to_string(myRank)+".dat");
  io_time =  (float)(clock() - timepoint) / CLOCKS_PER_SEC;

  float max_comp_time, total_io_time; 
  MPI_Reduce(&comp_time,&max_comp_time,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&io_time,&total_io_time,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

  if(myRank == 0)
   std::cout<<"Computaion(+comm)-time = "<<max_comp_time<<"s  IO-time = "<< total_io_time<<"s"<<std::endl;

  MPI_Finalize();
}


