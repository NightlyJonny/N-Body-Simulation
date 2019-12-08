#include "Drawer.h"

Drawer::Drawer() {

	int i = 0;
	glutInit(&i, NULL);
	
	quadratic = gluNewQuadric();
}

/*
x, y, z: The position vector
radius: Ray of sphere
angle: Angle of rotation expressed in radiant
xRot, yRot, zRot: The axis rotation vector
*/
void Drawer::drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLfloat angle, GLfloat xRot, GLfloat yRot, GLfloat zRot) {

	float height = radius/8;
	float ray = radius + 1E-2;

	//This will create a sphere and translate it
	glPushMatrix();
		
		glColor3f(1.0, 1.0, 1.0);
		glTranslatef(x, y, z);
		glRotatef(angle * 180 / M_PI, xRot, yRot, zRot);
		glutSolidSphere(radius, 16, 16);

		glColor3f(1.0, 0, 0);
		glRotatef(90, 1, 0, 0);
		gluCylinder(quadratic, ray, ray, height, 32, 1);
		glRotatef(90, 0, 1, 0);
		gluCylinder(quadratic, ray, ray, height, 32, 1);
	glPopMatrix();
}

void Drawer::drawSpaceship() {
	const GLfloat width = 0.3;
	const GLfloat length = 0.7;

	glBegin(GL_LINE_LOOP);
		glColor3f(1.0, 1.0, 1.0);
		glVertex2f(x, y);
		glVertex2f(x + width * cos(angle + 5*PI/8), y + width * sin(angle + 5*PI/8));
		glVertex2f(x + length * cos(angle), y + length * sin(angle));
		glVertex2f(x + width * cos(angle - 5*PI/8), y + width * sin(angle - 5*PI/8));
	glEnd();
}

void Drawer::setAsteroids(bool active) {

	asteroids = active;
}

void Drawer::rotateSpaceship(GLfloat da) {

	angle += da;
}

void Drawer::moveSpaceship(GLfloat ds) {

	x += ds * cos(angle);
	y += ds * sin(angle);
}

void Drawer::shoot() {

	glBegin(GL_LINES);
		glColor3f(1.0, 0, 0);
		glVertex2f(x, y);
		glVertex2f(x + 100 * cos(angle), y + 100 * sin(angle));
	glEnd();
}

void Drawer::draw_scene() {
	for (int i = 0; i < NParticle; i++) {
		if (!particle[i].active) continue;
		Vector3 pos = particle[i].position;
		Vector3 rotAxis = particle[i].omega.versor();
		drawSphere(pos.x, pos.y, pos.z, particle[i].radius, particle[i].angle, rotAxis.x, rotAxis.y, rotAxis.z);
	}
	if (asteroids) drawSpaceship();
	if (shooting) shoot();
}

void Drawer::setSimulation(Simulation* sim){

	this->particle = sim->getParticle();
	this->NParticle = sim->getNParticle();
}