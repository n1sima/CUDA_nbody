#include "bodysystem.h"
#include <iostream>
#include <vector>
#include <fstream>
#include "render_particles.h"


#include "GL/glew.h"
#include "GL/glut.h"



ParticleRenderer* renderer;
std::vector<float> positions;
std::vector<float> colors;
const int numBodies = 1024;

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderer->display(ParticleRenderer::PARTICLE_SPRITES_COLOR);
    glutSwapBuffers();
}


void idle() {
    glutPostRedisplay();
}



int main(int argc, char** argv) {

    const float clusterScale = 1.54f;
    const float velocityScale = 8.0f;

    positions.resize(numBodies * 4);
    colors.resize(numBodies * 4);
    std::vector<float> velocities(numBodies * 4);

    randomizeBodies<float>(
        NBODY_CONFIG_RANDOM,
        positions.data(),
        velocities.data(),
        colors.data(),
        clusterScale,
        velocityScale,
        numBodies,
        true // vec4vel
    );

    std::cout << "Generated " << numBodies << " particles:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Position: ("
                  << positions[i * 4] << ", "
                  << positions[i * 4 + 1] << ", "
                  << positions[i * 4 + 2] << ", "
                  << positions[i * 4 + 3] << ")\n";
    }

    // ✅ Save positions (x, y, z, w) to binary file
    std::ofstream outFile("positions.bin", std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open positions.bin for writing\n";
        return 1;
    }


    outFile.write(reinterpret_cast<char*>(positions.data()), positions.size() * sizeof(float));
    outFile.close();


    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Particle Renderer");

    glewInit(); // Needed if using GLEW

    renderer = new ParticleRenderer();
    renderer->setPositions(positions.data(), numBodies);
    renderer->setColors(colors.data(), numBodies);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMainLoop();

    delete renderer;

    std::cout << "Saved position data to positions.bin\n";
    return 0;
}
