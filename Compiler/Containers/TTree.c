#include <stdlib.h>
#include "$HEADER_NAME"

$BODY_PREAMBULE
inline static int ${NAME}_compareKeys($K lhs, $K rhs)
{
	return (lhs > rhs) - (lhs < rhs);
}


inline static ${NAME}_Node *${NAME}_createNode($K key, $V value)
{
	${NAME}_Node *node = (${NAME}_Node *)$MALLOC(sizeof(${NAME}_Node));

	node->key = key;
	node->value = value;

	node->links[0] = node->links[1] = NULL;
	node->red = true;

	return node;
}


inline static void ${NAME}_releaseNode(${NAME}_Node *node)
{
	$FREE(node);
}


inline static bool ${NAME}_nodeIsRed(${NAME}_Node *node)
{
	return (node != NULL) ? node->red : false;
}


inline static ${NAME}_Node *${NAME}_rotate(${NAME}_Node *node, size_t dir)
{
	${NAME}_Node *upnode = node->links[!dir];

	node->links[!dir] = upnode->links[dir];
	upnode->links[dir] = node;

	upnode->red = false;
	node->red = true;

	return upnode;
}


inline static ${NAME}_Node *${NAME}_rotate2(${NAME}_Node *node, size_t dir)
{
	${NAME}_Node *subnode = node->links[!dir];
	${NAME}_Node *upnode = subnode->links[dir];

	subnode->links[dir] = upnode->links[!dir];
	upnode->links[!dir] = subnode;

	node->links[!dir] = upnode->links[dir];
	upnode->links[dir] = node;

	upnode->red = false;
	node->red = true;

	return upnode;
}


void ${NAME}_init($NAME *self)
{
	self->root = NULL;
	self->size = 0;
}


void ${NAME}_dealloc($NAME *self)
{
	${NAME}_clear(self);
}


void ${NAME}_clear($NAME *self)
{
	${NAME}_Node *node = self->root;

	while (node != NULL)
	{
		${NAME}_Node *save = NULL;

		if (node->links[0] != NULL)
		{
			save = node->links[0];
			node->links[0] = save->links[1];
			save->links[1] = node;
		}
		else
		{
			save = node->links[1];
			${NAME}_releaseNode(node);
		}

		node = save;
	}

	self->size = 0;
	self->root = NULL;
}


inline static void ${NAME}_repairDoubleRed(${NAME}_Node *node, ${NAME}_Node *parent, ${NAME}_Node *grand,
										   ${NAME}_Node *temp, size_t lastdir)
{
	size_t dir2 = (temp->links[1] == grand);
	bool aligned = (node == parent->links[lastdir]);

	temp->links[dir2] = aligned ? ${NAME}_rotate(grand, !lastdir) : ${NAME}_rotate2(grand, !lastdir);
}


bool ${NAME}_insert($NAME *self, $K key, $V value)
{
	bool inserted = false;

	if (self->root != NULL)
	{
		${NAME}_Node head;

		head.red = false;
		head.links[0] = NULL;
		head.links[1] = self->root;

		${NAME}_Node *temp = &head;
		${NAME}_Node *parent = NULL, *grand = NULL;
		${NAME}_Node *node = self->root;

		size_t dir = 0, lastdir = 0;

		while (true)
		{
			if (node == NULL)
			{
				parent->links[dir] = node = ${NAME}_createNode(key, value);

				self->size += 1;
				inserted = true;
			}
			else if (${NAME}_nodeIsRed(node->links[0]) && ${NAME}_nodeIsRed(node->links[1]))
			{
				node->red = true;
				node->links[0]->red = node->links[1]->red = false;
			}

			if (${NAME}_nodeIsRed(node) && ${NAME}_nodeIsRed(parent))
				${NAME}_repairDoubleRed(node, parent, grand, temp, lastdir);

			int cmp = ${NAME}_compareKeys(key, node->key);
			if (cmp == 0)
				break;

			lastdir = dir;
			dir = cmp > 0;

			temp = (grand != NULL) ? grand : temp;

			grand = parent, parent = node;
			node = node->links[dir];
		}

		self->root = head.links[1];
	}
	else
	{
		self->root = ${NAME}_createNode(key, value);

		self->size = 1;
		inserted = true;
	}

	self->root->red = false;
	return inserted;
}


inline static ${NAME}_Node *${NAME}_repairDoubleBlack(${NAME}_Node *node, ${NAME}_Node *parent, ${NAME}_Node *grand,
													  size_t dir, size_t lastdir)
{
	if (${NAME}_nodeIsRed(node->links[!dir]))
	{
		parent = parent->links[lastdir] = ${NAME}_rotate(node, dir);
		return parent;
	}

	${NAME}_Node *sibling = parent->links[!lastdir];

	if (sibling != NULL)
	{
		if (!${NAME}_nodeIsRed(sibling->links[0]) && !${NAME}_nodeIsRed(sibling->links[1]))
		{
			parent->red = false;
			node->red = sibling->red = true;
		}
		else
		{
			size_t dir2 = (grand->links[1] == parent);

			if (${NAME}_nodeIsRed(sibling->links[lastdir]))
				grand->links[dir2] = ${NAME}_rotate2(parent, lastdir);

			else if (${NAME}_nodeIsRed(sibling->links[!lastdir]))
				grand->links[dir2] = ${NAME}_rotate(parent, lastdir);

			node->red = grand->links[dir2]->red = true;
			grand->links[dir2]->links[0]->red = grand->links[dir2]->links[1]->red = false;
		}
	}

	return parent;
}


bool ${NAME}_delete($NAME *self, $K key)
{
	if (self->root == NULL)
		return false;

	bool deleted = false;
	${NAME}_Node head;

	head.red = false;
	head.links[0] = NULL;
	head.links[1] = self->root;

	${NAME}_Node *found = NULL;
	${NAME}_Node *parent = NULL, *grand = NULL;
	${NAME}_Node *node = &head;

	size_t dir = 1;

	do
	{
		size_t lastdir = dir;

		grand = parent, parent = node;
		node = node->links[dir];

		int cmp = ${NAME}_compareKeys(key, node->key);

		dir = cmp > 0;
		found = (cmp == 0) ? node : found;

		if (!${NAME}_nodeIsRed(node) && !${NAME}_nodeIsRed(node->links[dir]))
			parent = ${NAME}_repairDoubleBlack(node, parent, grand, dir, lastdir);
	}
	while (node->links[dir] != NULL);

	if (found != NULL)
	{
		found->key = node->key;
		found->value = node->value;

		parent->links[parent->links[1] == node] = node->links[node->links[0] == NULL];

		${NAME}_releaseNode(node);

		self->size -= 1;
		deleted = true;
	}

	self->root = head.links[1];

	if (self->root != NULL)
		self->root->red = false;

	return deleted;
}


bool ${NAME}_get($NAME *self, $K key, $V *value)
{
	${NAME}_Node *node = self->root;

	while (node != NULL)
	{
		int cmp = ${NAME}_compareKeys(key, node->key);
		if (cmp == 0)
		{
			*value = node->value;
			return true;
		}
		else
			node = node->links[cmp > 0];
	}

	return false;
}


inline static bool ${NAME}_Iterator_start(${NAME}_Iterator *self, bool atLeft)
{
	size_t dir = atLeft ? 0 : 1;

	self->node = self->map->root;
	self->top = 0;

	if (self->node == NULL)
		return false;

	while (self->node->links[dir] != NULL)
	{
		self->path[self->top] = self->node;
		self->top += 1;

		self->node = self->node->links[dir];
	}

	return true;
}


bool ${NAME}_Iterator_init(${NAME}_Iterator *self, $NAME *map, bool atLeft)
{
	self->map = map;
	return ${NAME}_Iterator_start(self, atLeft);
}


void ${NAME}_Iterator_dealloc(${NAME}_Iterator *self)
{
	(void)self;
}


bool ${NAME}_Iterator_move(${NAME}_Iterator *self, bool toRight)
{
	size_t dir = toRight ? 1 : 0;

	if (self->node->links[dir] != NULL)
	{
		self->path[self->top] = self->node;
		self->top += 1;

		self->node = self->node->links[dir];

		while (self->node->links[!dir] != NULL)
		{
			self->path[self->top] = self->node;
			self->top += 1;

			self->node = self->node->links[!dir];
		}
	}
	else
	{
		${NAME}_Node *lastnode = NULL;
		do
		{
			if (self->top == 0)
			{
				self->node = NULL;
				break;
			}

			lastnode = self->node;

			self->top -= 1;
			self->node = self->path[self->top];
		}
		while (lastnode == self->node->links[dir]);
	}

	return self->node != NULL;
}


void ${NAME}_Iterator_item(${NAME}_Iterator *self, $K *key, $V *value)
{
	*key = self->node->key;
	*value = self->node->value;
}


static bool ${NAME}_validateNode(${NAME}_Node *node, size_t *nblack)
{
	size_t subblack[2] = {0, 0};

	for (size_t i = 0; i < 2; i += 1)
	{
		${NAME}_Node *subnode = node->links[i];
		if (subnode == NULL) continue;

		if (node->red && subnode->red)
			return false;

		if (!${NAME}_validateNode(subnode, &subblack[i]))
			return false;
	}

	if (subblack[0] != subblack[1])
		return false;

	size_t totalblack = subblack[0];
	if (!node->red) totalblack += 1;

	*nblack += totalblack;
	return true;
}


bool ${NAME}_validate($NAME *self)
{
	if (self->root == NULL)
		return true;

	size_t nblack = 0;
	return ${NAME}_validateNode(self->root, &nblack);
}
