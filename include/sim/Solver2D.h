#pragma once
#include "engine/Field.h"
// ---------------------------------------------------------------------------
// The 2D fluid solver.
//
// Owns the simulation state and, eventually, sequences the operators that act
// on it. Right now it only allocates - there is no physics yet.
// ---------------------------------------------------------------------------
class Solver2D
{
public:

	//The interface.
	void create(int width, int height);

	int width()  const { return m_width; }
	int height() const { return m_height; }

private:

	VectorField m_u;     // velocity - the actual simulation state
	ScalarField m_dye;   // dye - a passenger: visible, but physically inert

	int m_width = 0;
	int m_height = 0;
};