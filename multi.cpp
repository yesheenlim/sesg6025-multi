
#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

// definitions
#define ORDER 2 // order of stencil (only 2nd and 4th order implemented)
#define RB 0 // use Red-Black Gauss-Seidel? if not, SOR is used
#define MESH_SIZE 5 // dense matrix class use, mesh size cannot exceed 11 or be less than 3
#define GRID_SIZE (MESH_SIZE+2) // grid size
#define MAT_SIZE (MESH_SIZE*MESH_SIZE) // matrix size
#define STEP_SIZE (1.0/(GRID_SIZE-1)) // step size h
#define SOR_PARAM 1.4 // SOR parameter w

// namespaces
using namespace std;
using namespace boost;
using namespace Eigen;

// typedefs
typedef Matrix<double, Dynamic, Dynamic> M;
typedef Matrix<double, Dynamic, 1> V;

// check solution
shared_ptr<double> check(shared_ptr<V> x, shared_ptr<double> diag) {

    // displaying x in grid
    shared_ptr<M> Bt(new M), B(new M), U(new M);
    U->setZero(GRID_SIZE,GRID_SIZE);
    *Bt = Map<M>(x->data(),MESH_SIZE,MESH_SIZE);
    *B = Bt->transpose();
    U->block(1,1,MESH_SIZE,MESH_SIZE) = *B;
    // cout << "Values for u are: " << endl << *U << endl << endl;

    // check
    int c = (int)(GRID_SIZE/2);
    shared_ptr<double> check(new double(0));
    if(ORDER==2) {
        *check = ((*U)(c-1,c) + (*U)(c+1,c) +
                  (*U)(c,c-1) + (*U)(c,c-1) +
                  *diag*(*U)(c,c))/(STEP_SIZE*STEP_SIZE);

    } else if(ORDER==4) {
        *check = (-1.0*(*U)(c-2,c) + -1.0*(*U)(c+2,c) +
                  -1.0*(*U)(c,c-2) + -1.0*(*U)(c,c-2) +
                  16.0*(*U)(c-1,c) + 16.0*(*U)(c+1,c) +
                  16.0*(*U)(c,c-1) + 16.0*(*U)(c,c-1) +
                  *diag*(*U)(c,c))/(12.0*STEP_SIZE*STEP_SIZE);
    }

    return check;
}

// function to build E, y, x, and diag - 2nd order stencil
void build_Eyx_2(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> diag) {

    // build E
    E->setZero(MAT_SIZE,MAT_SIZE);
    int fc1=MESH_SIZE, sc1=0;
    for(int i=0; i<MAT_SIZE-1; i++) {
        // level 1 skip cascade
        if(sc1<MESH_SIZE-1) {
            (*E)(i+1,i) = 1;
            sc1++;
        } else {
            sc1 = 0;
        }
        // level 1 full cascade
        if(fc1<MAT_SIZE)
            (*E)(fc1++,i) = 1;
    }
    // cout << "E is: " << endl << *E << endl << endl;

    // build y
    y->setZero(MAT_SIZE,1);
    (*y)((int)(MAT_SIZE/2)) = 2.0*STEP_SIZE*STEP_SIZE;
    // cout << "The vector y is:" << endl << *y << endl << endl;

    // build x
    x->setZero(MAT_SIZE,1);
    // cout << "The initalised vector x is:" << endl << *x << endl << endl;

    // build diag
    *diag = -4;
}

// function to build E, y, x, and diag - 4th order stencil
void build_Eyx_4(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> diag) {

    // build E
    E->setZero(MAT_SIZE,MAT_SIZE);
    int fc1=MESH_SIZE, sc1=0, fc2=2*(MESH_SIZE-2)+4, sc2=0, sc2_=0;
    for(int i=0; i<MAT_SIZE-1; i++) {
        // level 1 skip cascade
        if(sc1<MESH_SIZE-1) {
            (*E)(i+1,i) = 16;
            sc1++;
        } else {
            sc1 = 0;
        }
        // level 1 full cascade
        if(fc1<MAT_SIZE)
            (*E)(fc1++,i) = 16;
        // level 2 skip cascade
        if(sc2<MESH_SIZE-2) {
            (*E)(i+2,i) = -1;
            sc2++;
        } else {
            if(sc2_>0) {
                sc2 = 0;
                sc2_ = 0;
            } else {
                sc2_++;
            }
        }
        // level 2 full cascade
        if(fc2<MAT_SIZE)
            (*E)(fc2++,i) = -1;
    }
    // cout << "E is: " << endl << *E << endl << endl;

    // build y
    y->setZero(MAT_SIZE,1);
    (*y)((int)(MAT_SIZE/2)) = 2.0*12.0*STEP_SIZE*STEP_SIZE;
    // cout << "The vector y is:" << endl << *y << endl << endl;

    // build x
    x->setZero(MAT_SIZE,1);
    // cout << "The initalised vector x is:" << endl << *x << endl << endl;

    // build diag
    *diag = -60;
    // cout << "Diagonal is constant " << *diag << endl << endl;
}

// function to build E, y, x, and diag - Red-Black Gauss-Seidel
void build_Eyx_RB(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> diag) {

    // note E is the upper right submatrix of Red-Black A
    // build E
    E->setZero((int)(MAT_SIZE/2)+1,(int)(MAT_SIZE/2));
    double m = MESH_SIZE-1;
    double m_ = MESH_SIZE-(0.5*m);

    int fc1=m_-1, fc2=0, sc1=(m/2),sc2=0;
    for(int i=0; i<(MAT_SIZE/2)+1; i++) {
        // full cascade 1
        if(fc1<(MAT_SIZE)/2)
            (*E)(i,fc1++) = 1;
        // full cascade 2
        if(i>=m_)
            (*E)(i,fc2++) = 1;
        // skip cascade 1
        if(i<(MAT_SIZE)/2) {
            if(sc1==m) {
                sc1=0;
            } else {
                sc1++;
                (*E)(i,i) = 1;
            }
        }
        // skip cascade 2
        if(i<(MAT_SIZE)/2) {
            if(sc2==m) {
                sc2 = 0;
            } else {
                sc2++;
                (*E)(i+1,i) = 1;
            }
        }
    }

    // build y (for the red nodes)
    y->setZero((MAT_SIZE/2)+1,1);
    (*y)((int)(y->size()/2)) = 2.0*STEP_SIZE*STEP_SIZE;
    // cout << "The vector y is:" << endl << *y << endl << endl;

    // build x
    x->setZero(MAT_SIZE,1);
    // cout << "The initalised vector x is:" << endl << *x << endl << endl;

    // build diag
    *diag = -4;
    // cout << "Diagonal is constant " << *diag << endl << endl;
}

// Red-Black Gauss-Seidel method to solve y=Ax
void solve_RBGS(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> diag) {

    shared_ptr<M> F(new M()); // F is the lower left submatrix of A
    shared_ptr<V> xr(new V()), xb(new V()), yb(new V());
    xr->setZero((int)((MAT_SIZE/2)+1),1);
    xb->setZero((int)(MAT_SIZE/2),1);
    yb->setZero((int)(MAT_SIZE/2),1);
    *F = E->transpose();

    for(int i=0; i<40; i++) {
        *xr = (-(*E)*(*xb) + *y)/(*diag);
        *xb = (-(*F)*(*xr) + *yb)/(*diag);
    }

    // cout << *xr << endl << endl;
    // cout << *xb << endl << endl;

    for(int i=1, p=0; i<x->size(); i+=2, p++){
	    (*x)(i-1) = (*xr)(p);
	    (*x)(i) = (*xb)(p);
    }
    (*x)(x->size()-1) = (*xr)(xr->size()-1);

    // check solution
    shared_ptr<double> rho = check(x,diag);
    cout << endl << "Check: rho(0.5,0.5) is " << *rho << endl << endl;
}

// SOR method to solve y=Ax
void solve_SOR(shared_ptr<M> E, shared_ptr<V> y, shared_ptr<V> x, shared_ptr<double> diag) {


    typedef Matrix<double, 1, Dynamic> V_;

    const double w(SOR_PARAM);
    const int m(MAT_SIZE);
    shared_ptr<V> exM;
    shared_ptr<V_> exM_;
    shared_ptr<V> exV;

    // SOR method a-go!
    // for each iteration do
    for(int k=0, i=0; k<40; k++, i=0) {

        // k stuff
        // extract part of matrix to multiply with xk
        exM.reset(new V);
        *exM = E->block(i+1,i,m-1-i,1);
        // extract xk
        exV.reset(new V);
        *exV = x->block(i+1,0,m-1-i,1);
        // compute
        (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - exM->dot(*exV))/(*diag) ) - (*x)(i) );

        for(i++; i<m-1; i++) {
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
            (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - sig)/(*diag) ) - (*x)(i) );
        }

        // k+1 stuff
        // extract part of matrix to multiply with xk+1
        exM_.reset(new V_);
        *exM_ = E->block(i,0,1,i);
        // extract xk+1
        exV.reset(new V);
        *exV = x->block(0,0,i,1);
        // compute
        (*x)(i) = (*x)(i) + w*( ( ((*y)(i) - exM_->dot(*exV))/(*diag) ) - (*x)(i) );

    }
    // cout << endl << "Ze solution iz: " << endl << *x << endl << endl;

    // check solution
    shared_ptr<double> rho = check(x,diag);
    cout << endl << "Check: rho(0.5,0.5) is " << *rho << endl << endl;
}

// execute solver
void execute() {

    // declare E, y, x, diag
    shared_ptr<double> diag(new double(0));
    shared_ptr<M> E(new M());
    shared_ptr<V> y(new V()), x(new V());

    // build E, x, y, diag
    if(RB==1) {
        build_Eyx_RB(E,y,x,diag);
    } else if(ORDER==2) {
        build_Eyx_2(E,y,x,diag);
    } else if(ORDER==4) {
        build_Eyx_4(E,y,x,diag);
    }

    // solve system
    if(RB==1) {
        solve_RBGS(E,y,x,diag);
    } else {
        solve_SOR(E,y,x,diag);
    }
}

// main
int main() {

    if(MESH_SIZE%2==0) {
        cout << endl << "The mesh size must be an odd number." << endl << endl;
        return 0;
    }
    if(ORDER!=2 && ORDER!=4) {
        cout << endl << "Only 2nd and 4th order stencil implemented." << endl << endl;
        return 0;
    }
    if(RB==1 && ORDER!=2) {
        cout << endl << "Red-black Gauss-Seidel only implemented for 2nd order stencil." << endl << endl;
        return 0;
    }

    execute();
}

