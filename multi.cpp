
#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
#include <Eigen/Dense>

// define constants
#define MESH_SIZE 3 // dense matrix class use, mesh size cannot exceed 11
#define GRID_SIZE (MESH_SIZE+2)
#define MAT_SIZE (MESH_SIZE*MESH_SIZE)
#define STEP_SIZE (1.0/(GRID_SIZE-1))
#define SOR_PARAM 1.4

// namespaces
using namespace std;
using namespace boost;
using namespace Eigen;

// typedefs
typedef Matrix<double, Dynamic, Dynamic> M;
typedef Matrix<double, Dynamic, 1> V;
typedef Matrix<double, 1, Dynamic> V_;

// check solution
shared_ptr<double> check(shared_ptr<V> x){

    // displaying x in grid
    shared_ptr<M> Bt(new M), B(new M), U(new M);
    U->setZero(GRID_SIZE,GRID_SIZE);
    *Bt = Map<M>(x->data(),MESH_SIZE,MESH_SIZE);
    *B = Bt->transpose();
    U->block(1,1,MESH_SIZE,MESH_SIZE) = *B;
//    cout << "Values for u are: " << endl << *U << endl << endl;

    // check
    int c = (int)(GRID_SIZE/2);
    double calc = ((*U)(c-1,c) + (*U)(c+1,c) +
                   (*U)(c,c-1) + (*U)(c,c-1)
                   - 4*(*U)(c,c))/(STEP_SIZE*STEP_SIZE);
    shared_ptr<double> check(new double(calc));

    return check;
}

// function to build E and y
void build_DEy(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x){

    // build F
    int c=MESH_SIZE, d=0;
    for(int i=0; i<MAT_SIZE-1; i++){
        if(d!=MESH_SIZE-1){
            (*E)(i+1,i) = 1;
            d++;
        }else{
            d = 0;
        }
        if(c<MAT_SIZE)
            (*E)(c++,i) = 1;
    }
    //    cout << "E is: " << endl << *E << endl << endl;

    // build y
    (*y)((int)(MAT_SIZE/2)) = 2.0*STEP_SIZE*STEP_SIZE;
    //    cout << "The vector y is:" << endl << *y << endl << endl;

    // build x
    x->setZero();
    //    cout << "The initalised vector x is:" << endl << *x << endl << endl;
}

// function to solve y=Ax
void solve(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> dc){

    const double w(SOR_PARAM);
    const int m(MAT_SIZE);
    shared_ptr<V> exM;
    shared_ptr<V_> exM_;
    shared_ptr<V> exV;

    // SOR method a-go!
    // for each iteration do
    for(int k=0, i=0; k<25; k++, i=0){

        // k stuff
        // extract part of matrix to multiply with xk
        exM.reset(new V);
        *exM = E->block(i+1,i,m-1-i,1);
        // extract xk
        exV.reset(new V);
        *exV = x->block(i+1,0,m-1-i,1);
        // compute
        (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - exM->dot(*exV))/(*dc) ) - (*x)(i) );

        for(i++; i<m-1; i++){
            // k+1 stuff
            // extract part of matrix to multiply with xk+1
            exM_.reset(new V_);
            *exM_ = E->block(i,0,1,i);
            // extract xk+1
            shared_ptr<V> exV_;
            exV_.reset(new V);
            *exV_ = x->block(0,0,i,1);

            // k stuff
            // extract part of matrix to multiply with xk
            exM.reset(new V);
            *exM = E->block(i+1,i,m-1-i,1);
            // extract xk
            exV.reset(new V);
            *exV = x->block(i+1,0,m-1-i,1);

            // compute
            const double sig = exM->dot(*exV) + exM_->dot(*exV_);
            (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - sig)/(*dc) ) - (*x)(i) );
        }

        // k+1 stuff
        // extract part of matrix to multiply with xk+1
        exM_.reset(new V_);
        *exM_ = E->block(i,0,1,i);
        // extract xk+1
        exV.reset(new V);
        *exV = x->block(0,0,i,1);
        // compute
        (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - exM_->dot(*exV))/(*dc) ) - (*x)(i) );

    }
    //    cout << endl << "Ze solution iz: " << endl << *x << endl << endl;
}

// execute solver
void execute(){
    // start timer
    //    timer t;

    // declare E, y, x
    shared_ptr<double> dc(new double(-4));
    shared_ptr<M> E(new M(MAT_SIZE, MAT_SIZE));
    shared_ptr<V> y(new V(MAT_SIZE,1)), x(new V(MAT_SIZE,1));

    // build F, x, y
    build_DEy(E,y,x);

    // solve system
    solve(E,y,x,dc);

    // check solution
    shared_ptr<double> rho = check(x);
    cout << endl << "Check: rho(0.5,0.5) is " << *rho << endl << endl;

    // time taken
    //    cout << endl << "CPU time taken: " << t.elapsed() << endl << endl;
}

// main
int main(){

    if(MESH_SIZE%2==0){
        cout << endl << "The mesh size must be an odd number." << endl << endl;
        return 0;
    }

    execute();
}
