
/* $Id$ */

#ifndef TYPE_H
#define TYPE_H

#ifdef FLOAT

typedef float Real;
#define _one_ 1.0f
#define _zero_ 0.0f
#define _pt5_ 0.5f
#define _four_ 4.0f

#else

typedef double Real;
#define _one_ 1.0
#define _zero_ 0.0
#define _pt5_ 0.5
#define _four_ 4.0

#endif

#define _NGrids_ 524288*16*255

#endif /* TYPE_H */
