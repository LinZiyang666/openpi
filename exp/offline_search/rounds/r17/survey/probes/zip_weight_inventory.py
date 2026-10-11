"""Read only ZIP central-directory metadata from public baseline weights."""
import urllib.request,struct,json,pathlib,datetime
root=pathlib.Path(__file__).resolve().parents[1]/'sources_v2'
rows=[]
for name,url in [('HULC_D_D','http://hulc.cs.uni-freiburg.de/model_weights/HULC_D_D.zip'),('HULC_ABC_D','http://hulc.cs.uni-freiburg.de/model_weights/HULC_ABC_D.zip'),('MCIL_D_D','http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip')]:
 try:
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=15) as r: size=int(r.headers['Content-Length'])
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={max(0,size-131072)}-{size-1}'}),timeout=15) as r:
   if r.status!=206:raise ValueError('server must support partial content; will not download checkpoint')
   data=r.read(131073)
   if len(data)>131072:raise ValueError('metadata limit exceeded')
  names=[]; pos=data.find(b'PK\x01\x02')
  while pos>=0 and data[pos:pos+4]==b'PK\x01\x02':
   n,e,c=struct.unpack_from('<HHH',data,pos+28)
   names.append({'path':data[pos+46:pos+46+n].decode(),'uncompressed_bytes':struct.unpack_from('<I',data,pos+24)[0]})
   pos+=46+n+e+c
  row={'name':name,'url':url,'zip_bytes':size,'metadata_download_bytes':len(data),'files':names,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 except Exception as e:row={'name':name,'url':url,'error':str(e)}
 rows.append(row);print(name, row.get('files',row))
(root/'calvin_zip_inventory.json').write_text(json.dumps(rows,indent=2))
