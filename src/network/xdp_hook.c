// ==============================================================================
// Next-GenAI: eBPF/XDP High-Speed Packet Hook
// Captures DoH (Port 443) traffic at line-rate before it hits the kernel stack.
// Language: C (Compiled via clang/LLVM)
// ==============================================================================
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>

#define MAX_PACKET_SIZE 1500

// BPF Map to send packets to userspace (Python)
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("xdp")
int xdp_doh_filter(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    // 1. Parse Ethernet Header
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != __constant_htons(ETH_P_IP)) return XDP_PASS;

    // 2. Parse IP Header
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;
    if (ip->protocol != IPPROTO_TCP) return XDP_PASS;

    // 3. Parse TCP Header (DoH runs over TCP 443)
    struct tcphdr *tcp = (void *)(ip + 1);
    if ((void *)(tcp + 1) > data_end) return XDP_PASS;

    // Check if destination or source port is 443 (HTTPS/DoH)
    if (tcp->dest == __constant_htons(443) || tcp->source == __constant_htons(443)) {
        // 4. Send packet payload to Python via RingBuffer
        void *payload = (void *)(tcp + 1);
        long len = data_end - payload;
        if (len > 0 && len < MAX_PACKET_SIZE) {
            void *ringbuf_slot = bpf_ringbuf_reserve(&rb, len, 0);
            if (ringbuf_slot) {
                bpf_probe_read_kernel(ringbuf_slot, len, payload);
                bpf_ringbuf_submit(ringbuf_slot, 0);
            }
        }
    }
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
