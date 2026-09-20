"""Unsettled owned purchases used to time corporate offsets against stock loans.

Spot sales use settled owned shares first, then the earliest outstanding receipt.
Cover purchases already committed to a loan return are not available owned shares.
This is scheduled regular-way custody, not a failed-delivery or locate simulator.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class ShareCustody:
    names: int
    receipts: list = field(default_factory=list)

    def settle(self, day):
        self.receipts = [(d, q) for d, q in self.receipts if d > day]

    def add(self, day, quantity):
        quantity = torch.as_tensor(quantity, dtype=torch.float64)
        if not bool((quantity.detach() > 0).any()):
            return
        for i, (due, previous) in enumerate(self.receipts):
            if due == day:
                self.receipts[i] = (day, previous + quantity)
                return
        self.receipts.append((day, quantity))
        self.receipts.sort(key=lambda item: item[0])

    def consume(self, held, quantity, day):
        """Remove owned shares, returning their actual custody-availability dates."""
        held = torch.as_tensor(held, dtype=torch.float64)
        quantity = torch.as_tensor(quantity, dtype=torch.float64)
        pending = sum((q for _, q in self.receipts), torch.zeros_like(held))
        settled = torch.minimum(quantity, (held - pending).clamp_min(0))
        used = [(day, settled)]
        remaining = (quantity - settled).clamp_min(0)
        receipts = []
        for due, receipt in self.receipts:
            take = torch.minimum(remaining, receipt)
            used.append((due, take))
            remaining = (remaining - take).clamp_min(0)
            remainder = (receipt - take).clamp_min(0)
            if bool((remainder.detach() > 0).any()):
                receipts.append((due, remainder))
        self.receipts = receipts
        return used

    def fill(self, held, purchases, sales, day, due):
        held = torch.as_tensor(held, dtype=torch.float64)
        sales = torch.as_tensor(sales, dtype=torch.float64)
        purchases = torch.as_tensor(purchases, dtype=torch.float64)
        self.consume(held, torch.minimum(sales, held), day)
        self.add(due, (purchases - (sales - held).clamp_min(0)).clamp_min(0))

    def split(self, name, ratio, *, bonus_delivery=None):
        adjusted = []
        bonus = []
        for due, quantity in self.receipts:
            changed = quantity.clone()
            if bonus_delivery is None:
                changed[name] = quantity[name] * ratio
            else:
                increment = torch.zeros_like(quantity)
                increment[name] = quantity[name] * (ratio - 1)
                bonus.append((max(due, bonus_delivery), increment))
            adjusted.append((due, changed))
        self.receipts = adjusted
        for due, increment in bonus:
            self.add(due, increment)

    def deliver(self, source, destination, ratio, incoming, existing, day, *, final):
        """Carry purchase value dates through succession, then reserve netted shares."""
        adjusted = []
        for due, quantity in self.receipts:
            changed = quantity.clone()
            changed[destination] = quantity[destination] + quantity[source] * ratio
            if final:
                changed[source] = 0
            adjusted.append((due, changed))
        self.receipts = adjusted
        incoming = torch.as_tensor(incoming, dtype=torch.float64)
        existing = torch.as_tensor(existing, dtype=torch.float64)
        held = torch.zeros(self.names, dtype=torch.float64)
        offset = torch.zeros_like(held)
        held[destination] = incoming.clamp_min(0) + existing.clamp_min(0)
        offset[destination] = torch.where(
            incoming * existing < 0, torch.minimum(incoming.abs(), existing.abs()), 0.0
        )
        # Other names' receipts are retained: only this successor is consumed.
        return self.consume(held, offset, day)

    def detach(self):
        self.receipts = [(day, quantity.detach()) for day, quantity in self.receipts]

    def detached_copy(self):
        return ShareCustody(
            self.names,
            [(day, quantity.detach().clone()) for day, quantity in self.receipts],
        )
