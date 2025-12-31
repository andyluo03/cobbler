#include "includes/controller.cuh"


int main () {
    // 1.
    auto controller = cobbler::Controller(); 
    
    // 2.
    controller.start(48);

    // // 3.
    // int taskOne = controller.dispatch();
    // int taskTwo = controller.dispatch();

    // // 4.
    // controller.await(taskOne);
    // controller.await(taskTwo);

    // // 5.
    // controller.end();
}