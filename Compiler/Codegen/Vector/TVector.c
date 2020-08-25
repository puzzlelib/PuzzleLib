#include <stdlib.h>
#include "$HEADER_NAME"

$BODY_PREAMBULE
void ${NAME}_init($NAME *self)
{
	self->ptr = NULL;
	self->size = self->capacity = 0;
}


void ${NAME}_dealloc($NAME *self)
{
	${NAME}_clear(self);
	$FREE(self->ptr);
}


void ${NAME}_reserve($NAME *self, size_t capacity)
{
	if (self->size < capacity)
	{
		$T *ptr = ($T *)$MALLOC(sizeof(self->ptr[0]) * capacity);

		for (size_t i = 0; i < self->size; i += 1)
			ptr[i] = self->ptr[i];

		$FREE(self->ptr);

		self->ptr = ptr;
		self->capacity = capacity;
	}
	else
	{
		for (size_t i = capacity; i < self->size; i += 1)
			$DESTRUCT(self->ptr[i]);

		self->size = self->capacity = capacity;
	}
}


inline static void ${NAME}_ensureIsAppendable($NAME *self)
{
	if (self->size == self->capacity)
	{
		size_t size = (self->capacity < $MIN_CAPACITY) ? $MIN_CAPACITY : self->capacity * 2;
		${NAME}_reserve(self, size);
	}
}


void ${NAME}_append($NAME *self, $T elem)
{
	${NAME}_ensureIsAppendable(self);

	$BORROW(elem);
	self->ptr[self->size] = elem;

	self->size += 1;
}


void ${NAME}_appendEmpty($NAME *self)
{
	${NAME}_ensureIsAppendable(self);
	self->size += 1;
}


bool ${NAME}_pop($NAME *self, $T *elem)
{
	if (self->size == 0)
		return false;

	self->size -= 1;
	*elem = self->ptr[self->size];

	return true;
}


void ${NAME}_clear($NAME *self)
{
	for (size_t i = 0; i < self->size; i += 1)
		$DESTRUCT(self->ptr[i]);

	self->size = 0;
}


bool ${NAME}_get($NAME *self, size_t index, $T *elem)
{
	if (index >= self->size)
		return false;

	*elem = self->ptr[index];
	return true;
}


bool ${NAME}_set($NAME *self, size_t index, $T elem)
{
	if (index >= self->size)
		return false;

	$BORROW(elem);
	$DESTRUCT(self->ptr[index]);

	self->ptr[index] = elem;
	return true;
}
