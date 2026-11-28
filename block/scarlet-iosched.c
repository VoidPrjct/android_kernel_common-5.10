// SPDX-License-Identifier: GPL-2.0
/*
 * Scarlet I/O scheduler - multiqueue hybrid for VoidPrjct
 *
 * Scarlet keeps the three ideas that pull their weight out of mq-deadline,
 * kyber and ADIOS, in one scheduler instead of letting three independent
 * schedulers compete for the same queue:
 *
 *   - Per-domain FIFOs (kyber). Requests are queued in one FIFO per domain:
 *     READ, WRITE, DISCARD, OTHER. The domain index is also the dispatch
 *     priority, so reads gate foreground apps first. A starve counter, taken
 *     from mq-deadline, forces a lower-priority turn so writes cannot be
 *     starved by a read flood.
 *
 *   - Deadline expiry and merging (mq-deadline). Every request carries an
 *     expiry in rq->fifo_time and the queue owns one timer armed to the
 *     earliest expiry, so queued I/O is looked at again even when no new
 *     request arrives to kick the queue. Merging uses the block layer request
 *     hash, exactly like mq-deadline, so adjacent bios coalesce.
 *
 *   - Latency-driven batching (kyber + ADIOS). Per-CPU latency histograms are
 *     folded into the batch size of each domain: when a domain misses its
 *     target the batch shrinks so other domains interleave sooner, and it
 *     grows back while the device keeps up. Async tag reservation follows
 *     kyber: two thirds of the queue is available to asynchronous I/O.
 *
 * No per-request allocation and no private trees are used: rq->fifo_time and
 * the block layer merge hash carry all the state. That keeps the failure
 * modes small, which matters for something that ships in a boot image.
 *
 * Copyright (C) 2026 VoidPrjct
 */

#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/elevator.h>
#include <linux/bio.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/compiler.h>
#include <linux/timer.h>
#include <linux/percpu.h>
#include <linux/math64.h>
#include <linux/kstrtox.h>
#include <linux/lockdep.h>

#include "blk-mq.h"
#include "blk-mq-sched.h"

#define SCARLET_VERSION "1.0.0"

/*
 * Scheduling domains. Index order is dispatch priority: when several domains
 * have work, the lower index wins. Reads first, they gate foreground apps.
 */
enum {
	SCARLET_READ,
	SCARLET_WRITE,
	SCARLET_DISCARD,
	SCARLET_OTHER,
	SCARLET_NUM_DOMAINS,
};

/* Only READ and WRITE carry latency samples. */
enum { SCARLET_LATENCY_DOMAINS = SCARLET_WRITE + 1 };

/* Latency targets: reads are latency critical, discard is background work. */
static const u64 scarlet_latency_targets[] = {
	[SCARLET_READ] = 2ULL * NSEC_PER_MSEC,
	[SCARLET_WRITE] = 10ULL * NSEC_PER_MSEC,
	[SCARLET_DISCARD] = 5ULL * NSEC_PER_SEC,
};

/* Upper bound of the adaptive batch size of each domain. */
static const unsigned int scarlet_max_batch[] = {
	[SCARLET_READ] = 16,
	[SCARLET_WRITE] = 8,
	[SCARLET_DISCARD] = 1,
	[SCARLET_OTHER] = 1,
};

/* Expiry in jiffies. Soft limits, same spirit as mq-deadline. */
static const int scarlet_read_expire = HZ / 2;
static const int scarlet_write_expire = 5 * HZ;

/* How many read batches may run before a lower-priority domain gets a turn. */
static const int scarlet_writes_starved = 2;

/*
 * Latency histogram relative to the target: 4 "good" buckets at or below the
 * target and 4 "bad" ones above it.
 */
enum {
	SCARLET_LATENCY_SHIFT = 2,
	SCARLET_GOOD_BUCKETS = 1 << SCARLET_LATENCY_SHIFT,
	SCARLET_LATENCY_BUCKETS = 2 << SCARLET_LATENCY_SHIFT,
};

struct scarlet_cpu_latency {
	atomic_t buckets[SCARLET_LATENCY_DOMAINS][SCARLET_LATENCY_BUCKETS];
};

struct scarlet_domain {
	struct list_head fifo;
};

struct scarlet_data {
	struct request_queue *queue;
	spinlock_t lock;
	struct scarlet_domain domains[SCARLET_NUM_DOMAINS];
	struct timer_list timer;
	unsigned int async_depth;
	unsigned int batch[SCARLET_NUM_DOMAINS];
	struct scarlet_cpu_latency __percpu *cpu_latency;
	unsigned long last_latency_update;
	unsigned int cur_domain;
	unsigned int batching;
	unsigned int starved;
};

/* ---- helpers ---- */

static unsigned int scarlet_domain_of_rq(struct request *rq)
{
	if (op_is_flush(rq->cmd_flags))
		return SCARLET_OTHER;
	if (op_is_discard(rq->cmd_flags))
		return SCARLET_DISCARD;
	/*
	 * op_is_write() also covers WRITE_ZEROES and ZONE_APPEND, which would
	 * otherwise be treated as reads and get read priority.
	 */
	if (op_is_write(req_op(rq)))
		return SCARLET_WRITE;
	return SCARLET_READ;
}

static unsigned long scarlet_expire_for(unsigned int domain)
{
	if (domain == SCARLET_READ)
		return jiffies + scarlet_read_expire;
	return jiffies + scarlet_write_expire;
}

/* ---- queue lifecycle ---- */

static void scarlet_timer_fn(struct timer_list *t)
{
	struct scarlet_data *sd = from_timer(sd, t, timer);

	blk_mq_run_hw_queues(sd->queue, true);
}

static int scarlet_init_sched(struct request_queue *q, struct elevator_type *e)
{
	struct scarlet_data *sd;
	struct elevator_queue *eq;
	int i;

	eq = elevator_alloc(q, e);
	if (!eq)
		return -ENOMEM;

	/*
	 * Zoned block devices need a scheduler that serialises writes per zone.
	 * Scarlet does not do that, so refuse the queue instead of letting
	 * writes be reordered into a zone that is still open.
	 */
	if (blk_queue_is_zoned(q)) {
		kobject_put(&eq->kobj);
		return -EINVAL;
	}

	sd = kzalloc_node(sizeof(*sd), GFP_KERNEL, q->node);
	if (!sd) {
		kobject_put(&eq->kobj);
		return -ENOMEM;
	}

	spin_lock_init(&sd->lock);
	sd->queue = q;
	sd->cur_domain = SCARLET_NUM_DOMAINS;

	sd->cpu_latency = alloc_percpu(struct scarlet_cpu_latency);
	if (!sd->cpu_latency)
		goto free_sd;

	for (i = 0; i < SCARLET_NUM_DOMAINS; i++) {
		INIT_LIST_HEAD(&sd->domains[i].fifo);
		sd->batch[i] = scarlet_max_batch[i];
	}

	/* Same split as kyber: two thirds of the tags for async I/O. */
	sd->async_depth = 2 * q->nr_requests / 3;
	if (!sd->async_depth)
		sd->async_depth = 1;
	sd->last_latency_update = jiffies;

	timer_setup(&sd->timer, scarlet_timer_fn, 0);

	eq->elevator_data = sd;
	q->elevator = eq;
	return 0;

free_sd:
	kfree(sd);
	kobject_put(&eq->kobj);
	return -ENOMEM;
}

static void scarlet_exit_sched(struct elevator_queue *e)
{
	struct scarlet_data *sd = e->elevator_data;

	del_timer_sync(&sd->timer);
	free_percpu(sd->cpu_latency);
	kfree(sd);
}

/* ---- tag reservation ---- */

static void scarlet_limit_depth(unsigned int op, struct blk_mq_alloc_data *data)
{
	struct scarlet_data *sd = data->q->elevator->elevator_data;

	/*
	 * Called with the queue locked: no sleeping here. Synchronous requests
	 * keep the whole queue; asynchronous ones are held to the same two
	 * thirds kyber reserves, so a writeback flood cannot take every tag.
	 */
	if (op_is_sync(op))
		return;

	data->shallow_depth = sd->async_depth;
}

/* ---- merging ---- */

static bool scarlet_bio_merge(struct request_queue *q, struct bio *bio,
			      unsigned int nr_segs)
{
	struct scarlet_data *sd = q->elevator->elevator_data;
	struct request *free = NULL;
	bool ret;

	spin_lock(&sd->lock);
	ret = blk_mq_sched_try_merge(q, bio, nr_segs, &free);
	spin_unlock(&sd->lock);

	if (free)
		blk_mq_free_request(free);

	return ret;
}

/*
 * @next has been merged into @rq and is about to be freed. Drop it from our
 * FIFO and from the merge hash so nothing can reference it again. When the
 * merge came from blk_mq_sched_try_insert_merge(), @next was never queued
 * here, and both calls below are no-ops.
 */
static void scarlet_requests_merged(struct request_queue *q,
				    struct request *rq, struct request *next)
{
	if (!list_empty(&next->queuelist))
		list_del_init(&next->queuelist);

	elv_rqhash_del(q, next);
	if (q->last_merge == next)
		q->last_merge = NULL;
}

/* ---- insert / dispatch ---- */

static void scarlet_insert_request(struct request_queue *q, struct request *rq,
				   bool at_head)
{
	struct scarlet_data *sd = q->elevator->elevator_data;
	unsigned int domain = scarlet_domain_of_rq(rq);
	struct scarlet_domain *dom = &sd->domains[domain];

	lockdep_assert_held(&sd->lock);

	/*
	 * rq->fifo_time is garbage on a fresh request (blk-mq only clears
	 * rq->deadline), so it must be written on every path. An at_head request
	 * is overdue by definition: expire it now so it is served first.
	 */
	if (at_head)
		rq->fifo_time = jiffies;
	else
		rq->fifo_time = scarlet_expire_for(domain);

	if (at_head)
		list_add(&rq->queuelist, &dom->fifo);
	else
		list_add_tail(&rq->queuelist, &dom->fifo);

	/*
	 * Hash every mergeable request so __blk_mq_sched_bio_merge() can find
	 * it. An at_head request may already be in the hash when it is
	 * reinserted after a failed dispatch, and elv_rqhash_add() BUG_ONs on
	 * that, so remove the stale entry first.
	 */
	if (rq_mergeable(rq)) {
		elv_rqhash_del(q, rq);
		elv_rqhash_add(q, rq);
		if (!q->last_merge)
			q->last_merge = rq;
	}

	if (!timer_pending(&sd->timer) ||
	    time_before((unsigned long)rq->fifo_time, sd->timer.expires))
		mod_timer(&sd->timer, (unsigned long)rq->fifo_time);

	blk_mq_sched_request_inserted(rq);
}

static void scarlet_insert_requests(struct blk_mq_hw_ctx *hctx,
				    struct list_head *list, bool at_head)
{
	struct request_queue *q = hctx->queue;
	struct scarlet_data *sd = q->elevator->elevator_data;

	spin_lock(&sd->lock);
	while (!list_empty(list)) {
		struct request *rq;

		rq = list_first_entry(list, struct request, queuelist);
		list_del_init(&rq->queuelist);

		/*
		 * A merged request is already freed inside this call, so it
		 * must not be queued or freed again.
		 */
		if (blk_mq_sched_try_insert_merge(q, rq))
			continue;

		scarlet_insert_request(q, rq, at_head);
	}
	spin_unlock(&sd->lock);
}

static bool scarlet_has_work(struct blk_mq_hw_ctx *hctx)
{
	struct request_queue *q = hctx->queue;
	struct scarlet_data *sd = q->elevator->elevator_data;
	int i;

	for (i = 0; i < SCARLET_NUM_DOMAINS; i++)
		if (!list_empty_careful(&sd->domains[i].fifo))
			return true;

	return false;
}

static struct request *scarlet_dequeue(struct scarlet_data *sd,
				       unsigned int domain)
{
	struct scarlet_domain *dom = &sd->domains[domain];
	struct request *rq;

	if (list_empty(&dom->fifo))
		return NULL;

	rq = list_first_entry(&dom->fifo, struct request, queuelist);
	list_del_init(&rq->queuelist);

	/* Same bookkeeping as mq-deadline's deadline_remove_request(). */
	elv_rqhash_del(sd->queue, rq);
	if (sd->queue->last_merge == rq)
		sd->queue->last_merge = NULL;

	return rq;
}

static struct request *scarlet_dispatch_from(struct scarlet_data *sd,
					     unsigned int *domain_out)
{
	struct request *rq;
	unsigned int domain;
	int i, start;

	lockdep_assert_held(&sd->lock);

	scarlet_latency_maybe_update(sd);

	/*
	 * Keep feeding the current domain until its batch is used up, so
	 * sequential I/O stays sequential on the device.
	 */
	if (sd->cur_domain < SCARLET_NUM_DOMAINS &&
	    sd->batching < sd->batch[sd->cur_domain]) {
		rq = scarlet_dequeue(sd, sd->cur_domain);
		if (rq) {
			sd->batching++;
			*domain_out = sd->cur_domain;
			return rq;
		}
	}

	/*
	 * Prefer READ, but once enough read batches have gone out, start the
	 * scan at WRITE so lower-priority domains cannot starve.
	 */
	if (sd->starved >= scarlet_writes_starved)
		start = SCARLET_WRITE;
	else
		start = SCARLET_READ;

	for (i = 0; i < SCARLET_NUM_DOMAINS; i++) {
		domain = (start + i) % SCARLET_NUM_DOMAINS;
		rq = scarlet_dequeue(sd, domain);
		if (!rq)
			continue;

		sd->cur_domain = domain;
		sd->batching = 1;
		if (domain == SCARLET_READ) {
			if (sd->starved < scarlet_writes_starved)
				sd->starved++;
		} else {
			sd->starved = 0;
		}

		*domain_out = domain;
		return rq;
	}

	return NULL;
}

static struct request *scarlet_dispatch_request(struct blk_mq_hw_ctx *hctx)
{
	struct request_queue *q = hctx->queue;
	struct scarlet_data *sd = q->elevator->elevator_data;
	struct request *rq;
	unsigned int domain;

	spin_lock(&sd->lock);
	rq = scarlet_dispatch_from(sd, &domain);
	spin_unlock(&sd->lock);

	return rq;
}

/* ---- latency sampling and adaptive batching ---- */

static void scarlet_latency_update(struct scarlet_data *sd)
{
	u64 good[SCARLET_LATENCY_DOMAINS] = {0};
	u64 bad[SCARLET_LATENCY_DOMAINS] = {0};
	int cpu, d, b;

	for_each_possible_cpu(cpu) {
		struct scarlet_cpu_latency *cl = per_cpu_ptr(sd->cpu_latency, cpu);

		for (d = 0; d < SCARLET_LATENCY_DOMAINS; d++) {
			for (b = 0; b < SCARLET_LATENCY_BUCKETS; b++) {
				int v = atomic_xchg(&cl->buckets[d][b], 0);

				if (b < SCARLET_GOOD_BUCKETS)
					good[d] += v;
				else
					bad[d] += v;
			}
		}
	}

	spin_lock(&sd->lock);
	for (d = 0; d < SCARLET_LATENCY_DOMAINS; d++) {
		u64 total = good[d] + bad[d];
		unsigned int nb = sd->batch[d];

		if (!total)
			continue;

		/*
		 * More than half the samples missed the target: shorten the
		 * batch so the other domains get a turn sooner. Otherwise grow
		 * back towards the maximum, which keeps sequential throughput.
		 */
		if (div64_u64(bad[d] * 100, total) > 50)
			nb = max_t(unsigned int, nb >> 1, 1);
		else if (nb < scarlet_max_batch[d])
			nb++;

		sd->batch[d] = nb;
	}
	spin_unlock(&sd->lock);
}

static void scarlet_completed_request(struct request *rq, u64 now)
{
	struct request_queue *q = rq->q;
	struct scarlet_data *sd = q->elevator->elevator_data;
	struct scarlet_cpu_latency *cl;
	unsigned int domain, bucket;
	u64 target, latency, divisor;

	domain = scarlet_domain_of_rq(rq);
	if (domain >= SCARLET_LATENCY_DOMAINS)
		return;
	if (!rq->io_start_time_ns)
		return;

	cl = get_cpu_ptr(sd->cpu_latency);
	target = scarlet_latency_targets[domain];
	latency = now - rq->io_start_time_ns;
	if (latency > 0) {
		divisor = max_t(u64, target >> SCARLET_LATENCY_SHIFT, 1);
		bucket = min_t(unsigned int, div64_u64(latency - 1, divisor),
			       SCARLET_LATENCY_BUCKETS - 1);
	} else {
		bucket = 0;
	}
	atomic_inc(&cl->buckets[domain][bucket]);
	put_cpu_ptr(cl);
}

/*
 * Fold the histograms and adapt the batch size under @sd->lock. Called from
 * the insertion path only: the completion path runs in softirq context, where
 * walking every CPU is too expensive.
 */
static void scarlet_latency_maybe_update(struct scarlet_data *sd)
{
	/* Fold at most every 100 ms. */
	if (!time_after(jiffies, sd->last_latency_update + HZ / 10))
		return;

	sd->last_latency_update = jiffies;
	scarlet_latency_update(sd);
}

/* ---- sysfs ---- */

#define SCARLET_BATCH_ATTRS(name, idx)					\
static ssize_t scarlet_##name##_batch_show(struct elevator_queue *e,	\
					   char *page)			\
{									\
	struct scarlet_data *sd = e->elevator_data;			\
	return sprintf(page, "%u\n", sd->batch[idx]);			\
}									\
static ssize_t scarlet_##name##_batch_max_show(struct elevator_queue *e,\
					       char *page)		\
{									\
	return sprintf(page, "%u\n", scarlet_max_batch[idx]);		\
}

SCARLET_BATCH_ATTRS(read, SCARLET_READ)
SCARLET_BATCH_ATTRS(write, SCARLET_WRITE)
SCARLET_BATCH_ATTRS(discard, SCARLET_DISCARD)
SCARLET_BATCH_ATTRS(other, SCARLET_OTHER)

static ssize_t scarlet_async_depth_show(struct elevator_queue *e, char *page)
{
	struct scarlet_data *sd = e->elevator_data;

	return sprintf(page, "%u\n", sd->async_depth);
}

static ssize_t scarlet_async_depth_store(struct elevator_queue *e,
					 const char *page, size_t count)
{
	struct scarlet_data *sd = e->elevator_data;
	unsigned int val;

	if (kstrtouint(page, 10, &val))
		return -EINVAL;
	/*
	 * Never below the queue's minimum shallow depth: the tag allocator warns
	 * (and then ignores the limit) for smaller values, which would stall
	 * asynchronous I/O instead of shaping it.
	 */
	if (val < 1 || val > sd->queue->nr_requests)
		return -EINVAL;
	val = min_t(unsigned int, val, sd->queue->nr_requests);
	sd->async_depth = val;
	return count;
}

static ssize_t scarlet_version_show(struct elevator_queue *e, char *page)
{
	return sprintf(page, "%s\n", SCARLET_VERSION);
}

static struct elv_fs_entry scarlet_attrs[] = {
	__ATTR(async_depth, 0644, scarlet_async_depth_show,
	       scarlet_async_depth_store),
	__ATTR(read_batch, 0444, scarlet_read_batch_show, NULL),
	__ATTR(write_batch, 0444, scarlet_write_batch_show, NULL),
	__ATTR(discard_batch, 0444, scarlet_discard_batch_show, NULL),
	__ATTR(other_batch, 0444, scarlet_other_batch_show, NULL),
	__ATTR(read_batch_max, 0444, scarlet_read_batch_max_show, NULL),
	__ATTR(write_batch_max, 0444, scarlet_write_batch_max_show, NULL),
	__ATTR(discard_batch_max, 0444, scarlet_discard_batch_max_show, NULL),
	__ATTR(other_batch_max, 0444, scarlet_other_batch_max_show, NULL),
	__ATTR(version, 0444, scarlet_version_show, NULL),
	__ATTR_NULL
};

static struct elevator_type mq_scarlet = {
	.ops = {
		.init_sched		= scarlet_init_sched,
		.exit_sched		= scarlet_exit_sched,
		.limit_depth		= scarlet_limit_depth,
		.bio_merge		= scarlet_bio_merge,
		.requests_merged	= scarlet_requests_merged,
		.insert_requests	= scarlet_insert_requests,
		.dispatch_request	= scarlet_dispatch_request,
		.completed_request	= scarlet_completed_request,
		.has_work		= scarlet_has_work,
	},
	.elevator_attrs		= scarlet_attrs,
	.elevator_name		= "scarlet",
	.elevator_features	= ELEVATOR_F_MQ_AWARE,
	.elevator_owner		= THIS_MODULE,
};
MODULE_ALIAS("mq-scarlet-iosched");

static int __init scarlet_init(void)
{
	return elv_register(&mq_scarlet);
}

static void __exit scarlet_exit(void)
{
	elv_unregister(&mq_scarlet);
}

module_init(scarlet_init);
module_exit(scarlet_exit);

MODULE_AUTHOR("VoidPrjct");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Scarlet hybrid I/O scheduler (mq-deadline + kyber + ADIOS ideas)");
