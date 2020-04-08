#pragma once

#include <stdbool.h>
#include <stdlib.h>


#if defined(ENABLE_TRACE_MALLOC)
	#define TRACE_MALLOC(size) ${NAME}_malloc(size, __FILE__, __LINE__)
	#define TRACE_FREE(ptr) ${NAME}_free(ptr)

#else
	#define TRACE_MALLOC(size) malloc(size)
	#define TRACE_FREE(ptr) free(ptr)

#endif


void *${NAME}_malloc(size_t size, const char *file, int line);
void ${NAME}_free(void *ptr);

size_t ${NAME}_traceLeaks(void);

bool ${NAME}_Iterator_init(void);
void ${NAME}_Iterator_dealloc(void);

bool ${NAME}_Iterator_move(void);
void ${NAME}_Iterator_item(size_t *size, const char **file, int *line);
