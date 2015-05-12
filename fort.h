

/* $Id$ */

#ifndef FMACRO_H
#define FMACRO_H

#ifdef NO_UNDERSCORE
#define FORT(x) x
#else
#define FORT(x) x##_
#endif

#endif /* FORT_H */
