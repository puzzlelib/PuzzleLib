#pragma once

#include <stddef.h>
#include <stdbool.h>

$HEADER_PREAMBULE
struct ${NAME}_Bucket;


typedef struct ${NAME}_Bucket
{
	$K key;
	$V value;

	struct ${NAME}_Bucket *next;
}
${NAME}_Bucket;


typedef struct $NAME
{
	${NAME}_Bucket **ptr;
	size_t size, log2capacity;
}
$NAME;


void ${NAME}_init($NAME *self);
void ${NAME}_dealloc($NAME *self);

bool ${NAME}_insert($NAME *self, $K key, $V value);
bool ${NAME}_delete($NAME *self, $K key);
bool ${NAME}_get($NAME *self, $K key, $V *value);
void ${NAME}_clear($NAME *self);
