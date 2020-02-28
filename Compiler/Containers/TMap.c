#include <stdlib.h>
#include "$HEADER_NAME"

$BODY_PREAMBULE
void ${NAME}_init($NAME *self)
{
	self->size = 0;
	self->log2capacity = $MIN_LOG2_CAPACITY;

	self->ptr = (${NAME}_Bucket **)$MALLOC(sizeof(self->ptr[0]) * (1ULL << self->log2capacity));

	for (size_t i = 0; i < (1ULL << self->log2capacity); i += 1)
		self->ptr[i] = NULL;
}


void ${NAME}_dealloc($NAME *self)
{
	${NAME}_clear(self);
	$FREE(self->ptr);
}


void ${NAME}_clear($NAME *self)
{
	for (size_t i = 0; i < (1ULL << self->log2capacity); i += 1)
	{
		${NAME}_Bucket *bucket = self->ptr[i];
		self->ptr[i] = NULL;

		while (bucket != NULL)
		{
			${NAME}_Bucket *next = bucket->next;

			$DESTRUCT_KEY(bucket->key);
			$DESTRUCT_VALUE(bucket->value);

			$FREE(bucket);
			bucket = next;
		}
	}
}


static void ${NAME}_rehash($NAME *self)
{
	size_t log2capacity = self->log2capacity + 1;
	${NAME}_Bucket **ptr = (${NAME}_Bucket **)$MALLOC(sizeof(self->ptr[0]) * (1ULL << log2capacity));

	for (size_t i = 0; i < (1ULL << log2capacity); i += 1)
		ptr[i] = NULL;

	for (size_t i = 0; i < (1ULL << self->log2capacity); i += 1)
	{
		${NAME}_Bucket *bucket = self->ptr[i];

		while (bucket != NULL)
		{
			size_t hash = $HASHER(bucket->key) & ((1 << log2capacity) - 1);
			${NAME}_Bucket **insert = &ptr[hash];

			while (*insert != NULL)
				insert = &(*insert)->next;

			*insert = bucket;

			${NAME}_Bucket *next = bucket->next;
			bucket->next = NULL;

			bucket = next;
		}
	}

	$FREE(self->ptr);
	self->ptr = ptr;

	self->log2capacity = log2capacity;
}


bool ${NAME}_insert($NAME *self, $K key, $V value)
{
	size_t hash = $HASHER(key) & ((1 << self->log2capacity) - 1);

	${NAME}_Bucket **ptr = &self->ptr[hash];
	${NAME}_Bucket *bucket = self->ptr[hash];

	while (bucket != NULL)
	{
		if ($COMPARE_KEYS(bucket->key, key))
		{
			bucket->value = value;
			return false;
		}

		ptr = &bucket->next;
		bucket = bucket->next;
	}

	${NAME}_Bucket *node = (${NAME}_Bucket *)$MALLOC(sizeof(*node));
	node->next = NULL;

	node->key = $BORROW_KEY(key);
	node->value = $BORROW_VALUE(value);

	*ptr = node;

	self->size += 1;
	float n = (float)self->size / (1 << self->log2capacity);

	if (n >= 0.75f)
		${NAME}_rehash(self);

	return true;
}


bool ${NAME}_delete($NAME *self, $K key)
{
	size_t hash = $HASHER(key) & ((1 << self->log2capacity) - 1);

	${NAME}_Bucket **ptr = &self->ptr[hash];
	${NAME}_Bucket *bucket = self->ptr[hash];

	while (bucket != NULL)
	{
		if ($COMPARE_KEYS(bucket->key, key))
		{
			*ptr = bucket->next;

			$DESTRUCT_KEY(bucket->key);
			$DESTRUCT_VALUE(bucket->value);

			$FREE(bucket);

			self->size -= 1;
			return true;
		}

		ptr = &bucket->next;
		bucket = bucket->next;
	}

	return false;
}


bool ${NAME}_get($NAME *self, $K key, $V *value)
{
	size_t hash = $HASHER(key) & ((1 << self->log2capacity) - 1);
	${NAME}_Bucket *bucket = self->ptr[hash];

	while (bucket != NULL)
	{
		if ($COMPARE_KEYS(bucket->key, key))
		{
			*value = bucket->value;
			return true;
		}

		bucket = bucket->next;
	}

	return false;
}
