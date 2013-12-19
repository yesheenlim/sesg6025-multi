
#include <iostream>
#include <iomanip>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
#include <Eigen/Dense>
#include <Eigen/LU>

// define constants
#define MESH_SIZE 7 // set the mesh size here
#define GRID_SIZE (MESH_SIZE+1)
#define MAT_SIZE (MESH_SIZE*MESH_SIZE)
#define STEP_SIZE (1.0/GRID_SIZE)

// namespaces
using namespace std;
using namespace boost;
using namespace Eigen;

// typedefs
typedef Matrix<double, MAT_SIZE, MAT_SIZE> M;
typedef Matrix<double, MESH_SIZE, MESH_SIZE> Msh;
typedef Matrix<double, MAT_SIZE, 1> V;
typedef FullPivLU<M> S;

// function to fill top/bottom right/left of A
void fill(shared_ptr<M> A){

    // fill interior of A
    for(int i=1; i<MESH_SIZE-1; i++){
        for(int j=1; j<MESH_SIZE-1; j++){
            int north = (i-1)*MESH_SIZE+j;
            int west = i*MESH_SIZE+j-1;
            int index = i*MESH_SIZE+j;
            int east = i*MESH_SIZE+j+1;
            int south = (i+1)*MESH_SIZE+j;

            (*A)(index,north) = 1;
            (*A)(index,west) = 1;
            (*A)(index,index) = -4;
            (*A)(index,east) = 1;
            (*A)(index,south) = 1;
        }
    }

    // fill top of A
    for(int j=1; j<MESH_SIZE-1; j++){
        int i=0;
        int west = i*MESH_SIZE+j-1;
        int index = i*MESH_SIZE+j;
        int east = i*MESH_SIZE+j+1;
        int south = (i+1)*MESH_SIZE+j;

        (*A)(index,west) = 1;
        (*A)(index,index) = -4;
        (*A)(index,east) = 1;
        (*A)(index,south) = 1;
    }

    // fill left of A
    for(int i=1; i<MESH_SIZE-1; i++){
        int j=0;
        int north = (i-1)*MESH_SIZE+j;
        int index = i*MESH_SIZE+j;
        int east = i*MESH_SIZE+j+1;
        int south = (i+1)*MESH_SIZE+j;

        (*A)(index,north) = 1;
        (*A)(index,index) = -4;
        (*A)(index,east) = 1;
        (*A)(index,south) = 1;
    }

    // fill right of A
    for(int i=1; i<MESH_SIZE-1; i++){
        int j = MESH_SIZE-1;
        int north = (i-1)*MESH_SIZE+j;
        int west = i*MESH_SIZE+j-1;
        int index = i*MESH_SIZE+j;
        int south = (i+1)*MESH_SIZE+j;

        (*A)(index,north) = 1;
        (*A)(index,west) = 1;
        (*A)(index,index) = -4;
        (*A)(index,south) = 1;
    }

    // fill bottom of A
    for(int j=1; j<MESH_SIZE-1; j++){
        int i = MESH_SIZE-1;
        int north = (i-1)*MESH_SIZE+j;
        int west = i*MESH_SIZE+j-1;
        int index = i*MESH_SIZE+j;
        int east = i*MESH_SIZE+j+1;

        (*A)(index,north) = 1;
        (*A)(index,west) = 1;
        (*A)(index,index) = -4;
        (*A)(index,east) = 1;
    }

    int i_, j_, north_, east_, south_, west_, index_;

    // fill top left of A
    i_=0, j_=0;
    index_ = i_*MESH_SIZE+j_;
    east_ = i_*MESH_SIZE+j_+1;
    south_ = (i_+1)*MESH_SIZE+j_;
    (*A)(index_,index_) = -4;
    (*A)(index_,east_) = 1;
    (*A)(index_,south_) = 1;

    // fill top right of A
    i_=0, j_=MESH_SIZE-1;
    west_ = i_*MESH_SIZE+j_-1;
    index_ = i_*MESH_SIZE+j_;
    south_ = (i_+1)*MESH_SIZE+j_;
    (*A)(index_,west_) = 1;
    (*A)(index_,index_) = -4;
    (*A)(index_,south_) = 1;

    // fill bottom left of A
    i_=MESH_SIZE-1, j_=0;
    north_ = (i_-1)*MESH_SIZE+j_;
    index_ = i_*MESH_SIZE+j_;
    east_ = i_*MESH_SIZE+j_+1;
    (*A)(index_,north_) = 1;
    (*A)(index_,index_) = -4;
    (*A)(index_,east_) = 1;

    // fill bottom right of A
    i_=MESH_SIZE-1, j_=MESH_SIZE-1;
    north_ = (i_-1)*MESH_SIZE+j_;
    west_ = i_*MESH_SIZE+j_-1;
    index_ = i_*MESH_SIZE+j_;
    (*A)(index_,north_) = 1;
    (*A)(index_,west_) = 1;
    (*A)(index_,index_) = -4;

}

// function to build A
void build_A(shared_ptr<M> A){

    fill(A);
    cout << "The matrix A is:" << endl << *A << endl << endl;
}

// function to build y
void build_y(shared_ptr<V> y){

    (*y)[(int)(MAT_SIZE/2)] = 2.0*STEP_SIZE*STEP_SIZE;

    cout << "The vector y is:" << endl << *y << endl << endl;

}

// function to solve y=Ax
void solve(shared_ptr<M> A, shared_ptr<V> y, shared_ptr<V> x){


    // too store output grid
    shared_ptr<Msh> Bt(new Msh), B(new Msh);

    // use Eigen's LU decomposition solver
    shared_ptr<S > solver(new S(*A));

    *x = solver->solve(*y);
    cout << "Solution for x is: " << endl << *x << endl << endl;

    // displaying x in grid
    *Bt = Map<Msh>(x->data(),MESH_SIZE,MESH_SIZE);
    *B = Bt->transpose();
    cout << "Values for u are: " << endl << *B << endl << endl;

    int c = (int)(MESH_SIZE/2);
    double check = ((*B)(c-1,c) + (*B)(c+1,c) +
                    (*B)(c,c-1) + (*B)(c,c-1)
                    - 4*(*B)(c,c))/(STEP_SIZE*STEP_SIZE);
    cout << "Checking: rho(0.5,0.5) = " << check << endl << endl;

}


// main
int main(){

    //start timer
    timer t;

    if(MESH_SIZE%2==0){
        cout << endl << "The mesh size must be an odd number." << endl << endl;
        return 0;
    }
    //pointer to A and y
    shared_ptr<M> A(new M);
    shared_ptr<V> y(new V);
    shared_ptr<V> x(new V);

    //build A and y
    build_A(A);
    build_y(y);

    //solve system
    solve(A,y,x);

    //time taken
    cout << "CPU time taken: " << t.elapsed() << endl << endl;

}
