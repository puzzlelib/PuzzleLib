#pragma once

#include <stddef.h>
#include <stdbool.h>

$HEADER_PREAMBULE
typedef struct $NAME
{
	$T *ptr;
	size_t size, capacity;
}
$NAME;


void ${NAME}_init($NAME *self);
void ${NAME}_dealloc($NAME *self);

void ${NAME}_resize($NAME *self, size_t capacity);
void ${NAME}_append($NAME *self, $T elem);
bool ${NAME}_pop($NAME *self, $T *elem);
void ${NAME}_clear($NAME *self);
bool ${NAME}_get($NAME *self, size_t index, $T *elem);
bool ${NAME}_set($NAME *self, size_t index, $T elem);
