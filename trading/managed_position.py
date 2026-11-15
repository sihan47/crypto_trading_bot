"""Confirmed strategy-owned lots, separate from unrelated wallet holdings."""
from decimal import Decimal, InvalidOperation, ROUND_DOWN


def amount(value):
    try:
        result=Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid amount') from exc
    if not result.is_finite() or result<0:
        raise ValueError('Amount must be finite and nonnegative')
    return result


def managed_position(journal,key,mode='live'):
    state=journal._read()
    symbol=key.split(':')[0]
    # A different/older strategy's ambiguous request must also be reconciled.
    if any(k.split(':')[0]==symbol and v.get('pending') for k,v in state['orders'].items()):
        raise ValueError('Unresolved exchange order; reconcile before evaluating this asset')
    if mode=='shadow':
        return state.get('shadow',{}).get(key,{}).get('side')=='BUY',Decimal('0')
    if any(k != key and k.split(':')[0] == symbol and amount(v.get('managed_qty','0')) > 0
           for k,v in state['orders'].items()):
        raise ValueError('Another strategy owns a live lot; reconcile before switching engines')
    record=state['orders'].get(key,{})
    qty=amount(record.get('managed_qty','0'))
    return qty>0,qty


def market_quantity(info,desired,reference_price):
    """Round down and validate LOT_SIZE/MARKET_LOT_SIZE and market notionals."""
    if not info or info.get('status')!='TRADING':
        raise ValueError('Symbol is not trading')
    qty=amount(desired);price=amount(reference_price)
    filters=info.get('filters',[])
    lots=[f for f in filters if f['filterType'] in {'LOT_SIZE','MARKET_LOT_SIZE'}]
    if not any(f['filterType']=='LOT_SIZE' for f in lots):
        raise ValueError('Missing lot-size rules')
    for f in lots:
        step=amount(f['stepSize'])
        if step:
            qty=(qty/step).to_integral_value(rounding=ROUND_DOWN)*step
    if qty<=0 or price<=0:
        raise ValueError('Quantity/price is zero')
    for f in lots:
        step=amount(f['stepSize']);minimum=amount(f['minQty']);maximum=amount(f['maxQty'])
        if qty<minimum or (maximum and qty>maximum) or (step and qty%step):
            raise ValueError('Quantity fails market lot rules')
    for f in filters:
        if f['filterType']=='MIN_NOTIONAL' and f.get('applyToMarket',True):
            if qty*price<amount(f['minNotional']):
                raise ValueError('Quantity below minimum notional')
        if f['filterType']=='NOTIONAL':
            if f.get('applyMinToMarket',False) and qty*price<amount(f['minNotional']):
                raise ValueError('Quantity below minimum notional')
            if f.get('applyMaxToMarket',False) and qty*price>amount(f['maxNotional']):
                raise ValueError('Quantity above maximum notional')
    return format(qty,'f')


def confirm_managed_order(journal,key,order,base,requested,side):
    """Keep pending intent on incomplete/inconsistent responses; never guess fills."""
    if not isinstance(order,dict) or not order.get('orderId') or order.get('status')!='FILLED':
        raise ValueError('Only a FILLED order can update a managed lot; reconcile pending order')
    executed=amount(order.get('executedQty','0'))
    if not executed or executed!=amount(requested) or order.get('side')!=side:
        raise ValueError('Unexpected filled quantity/side; reconcile pending order')
    fills=order.get('fills')
    if not isinstance(fills,list) or not fills or sum(amount(f['qty']) for f in fills)!=executed:
        raise ValueError('Complete execution and commission details required')
    fee=sum(amount(f['commission']) for f in fills if f['commissionAsset']==base)
    state=journal._read();record=state['orders'][key]
    if record.get('pending',{}).get('side')!=side:
        raise ValueError('Fill does not match persisted intent')
    old=amount(record.get('managed_qty','0'))
    if side=='BUY':
        if old or fee>=executed:
            raise ValueError('Unexpected existing position or commission')
        record['managed_qty']=str(executed-fee)
    else:
        remaining=old-executed-fee
        if remaining<0:
            raise ValueError('Sell consumed more than the strategy-owned position')
        # Caller requested the full rounded lot. Record the untradable remainder
        # separately instead of treating wallet dust as an active long forever.
        record['dust_qty']=str(amount(record.get('dust_qty','0'))+remaining)
        record['managed_qty']='0'
    record['last_side']=side
    record['order_id']=order['orderId']
    record.pop('pending')
    journal._write(state)
