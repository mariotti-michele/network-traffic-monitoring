module Tensorflow;
# definisce un nome per l'evento generico e quali sono i dati raccolti
global content: event(orig_h:addr, orig_p:port, resp_h:addr, resp_p:port, content:string);

# inizializza il broker
event zeek_init()
{
  Broker::subscribe("tensorflow/content");
  Broker::listen("192.168.1.76", 9999/tcp);
  Broker::auto_publish("tensorflow/content", content);
}

# funzione che pubblica l’evento "content", contenente informazioni sulla connessione
function output(c:connection, bytes:string)
{
  Broker::publish("tensorflow/content", content, c$id$orig_h,
  c$id$orig_p, c$id$resp_h, c$id$resp_p, bytes);
}

# analizza i pacchetti TCP e, se il pacchetto è il primo della connessione (SYN, seq=1, con payload), lo invia
event tcp_packet(c:connection, is_orig:bool, flags:string, seq:count, ack:count, len:count, payload:string)
{
  if(is_orig && "S" in c$history && seq==1 && |payload|>0)
  {
    output(c, string_to_ascii_hex(payload));
  }
}
