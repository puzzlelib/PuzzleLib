#pragma once

#include <stddef.h>
#include <stdbool.h>

$HEADER_PREAMBULE
struct ${NAME}_Node;


typedef struct ${NAME}_Node
{
	bool red;
	struct ${NAME}_Node *links[2];

	$K key;
	$V value;
}
${NAME}_Node;


typedef struct $NAME
{
	${NAME}_Node *root;
	size_t size;
}
$NAME;


void ${NAME}_init($NAME *self);
void ${NAME}_dealloc($NAME *self);
bool ${NAME}_validate($NAME *self);

bool ${NAME}_insert($NAME *self, $K key, $V value);
bool ${NAME}_delete($NAME *self, $K key);
bool ${NAME}_get($NAME *self, $K key, $V *value);
void ${NAME}_clear($NAME *self);


typedef struct ${NAME}_Iterator
{
	$NAME *map;
	${NAME}_Node *node;

	${NAME}_Node *path[16 * sizeof(size_t)];
	size_t top;
}
${NAME}_Iterator;


bool ${NAME}_Iterator_init(${NAME}_Iterator *self, $NAME *map, bool atLeft);
void ${NAME}_Iterator_dealloc(${NAME}_Iterator *self);

bool ${NAME}_Iterator_move(${NAME}_Iterator *self, bool toRight);
void ${NAME}_Iterator_item(${NAME}_Iterator *self, $K *key, $V *value);
